#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/count.hpp>

#include "optimization_commons.h"

#include "haarwavelet.h"
#include "haarwaveletutilities.h"
#include "haarwaveletevaluators.h"

#include "sampleextractor.h"

#include <tbb/tbb.h>



//http://www.boost.org/doc/libs/1_53_0/doc/html/accumulators/user_s_guide.html
typedef boost::accumulators::accumulator_set<double,
                                             boost::accumulators::stats<boost::accumulators::tag::mean,
                                                                        boost::accumulators::tag::variance,
                                                                        boost::accumulators::tag::count> > myaccumulator;



void produceFeatureValues(myaccumulator & acc,
                          const HaarWavelet & wavelet,
                          const std::vector<cv::Mat> & integralSums)
{
    const cv::Mat uselessMat;

    const IntensityNormalizedWaveletEvaluator evaluator;

    const unsigned int records = integralSums.size();
    for (unsigned int i = 0; i < records; ++i)
    {
        acc( evaluator(wavelet, integralSums[i], uselessMat) );
    }
}



/**
 * Stores data to be used in weak classifiers that operate like Adhikari's paper
 * "Boosting-Based On-Road Obstacle Sensing Using Discriminative Weak Classifiers".
 *
 * IMPORTANT: this program trains Adhikari's classifier with the intensity normalization
 * of the sample images.
 */
class ProbabilisticClassifierData : public HaarWavelet
{
public:
    ProbabilisticClassifierData() : positiveMean(.0), positiveVariance(1.0),
                                    negativeMean(.0), negativeVariance(1.0) {}

    ProbabilisticClassifierData(const HaarWavelet & wavelet) : HaarWavelet(wavelet),
                                                               positiveMean(.0), positiveVariance(1.0),
                                                               negativeMean(.0), negativeVariance(1.0) {}

    ProbabilisticClassifierData& operator=(const ProbabilisticClassifierData & c)
    {
        rects = c.rects;
        weights = c.weights;

        positiveMean = c.positiveMean;
        positiveVariance = c.positiveVariance;
        negativeMean = c.negativeMean;
        negativeVariance = c.negativeVariance;

        positiveSamplesCount = c.positiveSamplesCount;
        negativeSamplesCount = c.negativeSamplesCount;

        return *this;
    }

    void setPositiveVariance(const double var)
    {
        positiveVariance = var;
    }

    void setPositiveMean(const double mean_)
    {
        positiveMean = mean_;
    }

    void setNegativeVariance(const double var)
    {
        negativeVariance = var;
    }

    void setNegativeMean(const double mean_)
    {
        negativeMean = mean_;
    }

    void setPositiveSamplesCount(int c)
    {
        positiveSamplesCount = c;
    }

    void setNegativeSamplesCount(int c)
    {
        negativeSamplesCount = c;
    }

    bool operator < (const ProbabilisticClassifierData & rh) const
    {
        return positiveVariance < rh.positiveVariance;
    }

    bool write(std::ostream &output) const
    {
        if ( !HaarWavelet::write(output) )
        {
            return false;
        }

        output << ' '
               << positiveMean << ' '
               << positiveVariance << ' '
               << (positiveSamplesCount / (positiveSamplesCount + negativeSamplesCount)) << ' '
               << negativeMean << ' '
               << negativeVariance << ' '
               << (negativeSamplesCount / (positiveSamplesCount + negativeSamplesCount));

        return true;
    }

private:
     //statistics taken from the feature value, not directly from the SRFS
    double positiveMean, positiveVariance;
    double negativeMean, negativeVariance;

    double positiveSamplesCount, negativeSamplesCount;
};



/**
 * Functor used by Intel TBB to optimize the haar-like feature based classifiers using PCA.
 */
class Optimize
{
private:
    std::vector<HaarWavelet> * wavelets;
    std::vector<cv::Mat> * positivesIntegralSums;
    std::vector<cv::Mat> * negativesIntegralSums;
    tbb::concurrent_vector<ProbabilisticClassifierData> * classifiers;



public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ProbabilisticClassifierData classifier( (*wavelets)[i] );

            {
                myaccumulator acc;
                produceFeatureValues(acc, classifier, *positivesIntegralSums);
                classifier.setPositiveMean(boost::accumulators::mean(acc));
                classifier.setPositiveVariance(boost::accumulators::variance(acc));
                classifier.setPositiveSamplesCount(boost::accumulators::count(acc));
            }

            {
                myaccumulator acc;
                produceFeatureValues(acc, classifier, *negativesIntegralSums);
                classifier.setNegativeMean(boost::accumulators::mean(acc));
                classifier.setNegativeVariance(boost::accumulators::variance(acc));
                classifier.setNegativeSamplesCount(boost::accumulators::count(acc));
            }

            classifiers->push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet> * wavelets_,
             std::vector<cv::Mat> * positivesIntegralSums_,
             std::vector<cv::Mat> * negativesIntegralSums_,
             tbb::concurrent_vector<ProbabilisticClassifierData> * classifiers_) : wavelets(wavelets_),
                                                                                   positivesIntegralSums(positivesIntegralSums_),
                                                                                   negativesIntegralSums(negativesIntegralSums_),
                                                                                   classifiers(classifiers_) {}
};



void writeClassifiersData(std::ofstream & outputStream, tbb::concurrent_vector<ProbabilisticClassifierData> & classifiers)
{
    tbb::concurrent_vector<ProbabilisticClassifierData>::const_iterator it = classifiers.begin();
    tbb::concurrent_vector<ProbabilisticClassifierData>::const_iterator end = classifiers.end();
    for(; it != end; ++it)
    {
        it->write(outputStream);
        outputStream << '\n';
    }
}



/**
 * Loads the Haar wavelets from a file and the image samples found in a directory, then produce
 * the SRFS for each Haar wavelet. Extract the principal component of least variance and use it
 * as the new weights of the respective Haar wavelet. When all is done, write the 'optimized'
 * Haar wavelets to a file.
 */
int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        std::cout << "Usage " << argv[0] << " " << " WAVELETS_FILE POSITIVE_SAMPLES_FILE NEGATIVE_SAMPLES_FILE NEGATIVE_SAMPLES_INDEX OUTPUT_DIR" << std::endl;
        return 1;
    }

    const std::string waveletsFileName     = argv[1]; //load Haar wavelets from here
    const std::string positiveSamplesImage = argv[2]; //load + samples from here
    const std::string negativeSamplesImage = argv[3]; //load - samples from here
    const std::string negativeSamplesIndex = argv[4]; //load - samples from here
    const std::string classifiersFileName  = argv[5]; //write output here



    std::vector<HaarWavelet> wavelets;
    std::vector<cv::Mat> positivesIntegralSums, negativesIntegralSums;
    std::ofstream outputStream;


    {
        //Load a list of Haar wavelets
        std::cout << "Loading wavelets..." << std::endl;
        if (!loadHaarWavelets(waveletsFileName, wavelets))
        {
            std::cout << "Unable to load Haar wavelets from file " << waveletsFileName << std::endl;
            return 2;
        }
        std::cout << wavelets.size() << " wavelets loaded." << std::endl;

        outputStream.open(classifiersFileName.c_str(), std::ios::trunc);
        if ( !outputStream.is_open() )
        {
            std::cout << "Can't open output file." << std::endl;
            return 5;
        }

        //TODO The second parameter here does not match what is expected for the rest of the program!
        if ( !SampleExtractor::extractFromBigImage(positiveSamplesImage, positivesIntegralSums) )
        {
            std::cout << "Failed to load positive samples." << std::endl;
            return 6;
        }
        std::transform(positivesIntegralSums.begin(), positivesIntegralSums.end(),
                       positivesIntegralSums.begin(),
                       ToIntegralSums());
        std::cout << positivesIntegralSums.size() << " positive samples loaded." << std::endl;

        //TODO The third parameter here does not match what is expected for the rest of the program!
        if ( !SampleExtractor::extractFromBigImage(negativeSamplesImage, negativeSamplesIndex, negativesIntegralSums) )
        {
            std::cout << "Failed to load negative samples." << std::endl;
            return 7;
        }
        std::transform(negativesIntegralSums.begin(), negativesIntegralSums.end(),
                       negativesIntegralSums.begin(),
                       ToIntegralSums());
        std::cout << negativesIntegralSums.size() << " negative samples loaded." << std::endl;
    }



    std::cout << "Optimizing Haar-like features..." << std::endl;

    tbb::concurrent_vector<ProbabilisticClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &positivesIntegralSums, &negativesIntegralSums, &classifiers));



    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());

    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;

    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
