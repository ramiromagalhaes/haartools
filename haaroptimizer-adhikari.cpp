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



struct Integrals
{
    cv::Mat iSum, iSquare;

    Integrals() {}

    Integrals(cv::Mat & iSum_, cv::Mat & iSquare_)
    {
        iSum = iSum_;
        iSquare = iSquare_;
    }

    Integrals & operator=(const Integrals & i)
    {
        iSum = i.iSum;
        iSquare = i.iSquare;
        return *this;
    }
};



struct ToIntegrals
{
    inline Integrals operator()(cv::Mat & image) const
    {
        cv::Mat iSum(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::Mat iSquare(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::integral(image, iSum, iSquare, cv::DataType<double>::type);

        Integrals i(iSum, iSquare);
        return i;
    }
};



void produceFeatureValues(myaccumulator & acc,
                          const HaarWavelet & wavelet,
                          const std::vector<Integrals> & integrals)
{
    const VarianceNormalizedWaveletEvaluator evaluator;

    const unsigned int records = integrals.size();
    for (unsigned int i = 0; i < records; ++i)
    {
        acc( evaluator(wavelet, integrals[i].iSum, integrals[i].iSquare) );
    }
}



/**
 * Stores data to be used in weak classifiers that operate like Adhikari's paper
 * "Boosting-Based On-Road Obstacle Sensing Using Discriminative Weak Classifiers".
 *
 * IMPORTANT: this program trains Adhikari's classifier with the variance normalization
 * of the sample images. Although Adhikari's paper does not mention what normalization
 * procedure they used, I believe this was the chosen one.
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
    std::vector<HaarWavelet> & wavelets;
    std::vector<Integrals> & positiveIntegrals;
    std::vector<Integrals> & negativeIntegrals;
    tbb::concurrent_vector<ProbabilisticClassifierData> & classifiers;



public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ProbabilisticClassifierData classifier( wavelets[i] );

            {
                myaccumulator acc;
                produceFeatureValues(acc, classifier, positiveIntegrals);
                classifier.setPositiveMean(boost::accumulators::mean(acc));
                classifier.setPositiveVariance(boost::accumulators::variance(acc));
                classifier.setPositiveSamplesCount(boost::accumulators::count(acc));
            }

            {
                myaccumulator acc;
                produceFeatureValues(acc, classifier, negativeIntegrals);
                classifier.setNegativeMean(boost::accumulators::mean(acc));
                classifier.setNegativeVariance(boost::accumulators::variance(acc));
                classifier.setNegativeSamplesCount(boost::accumulators::count(acc));
            }

            classifiers.push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet> & wavelets_,
             std::vector<Integrals> & positiveIntegrals_,
             std::vector<Integrals> & negativeIntegrals_,
             tbb::concurrent_vector<ProbabilisticClassifierData> & classifiers_) : wavelets(wavelets_),
                                                                                   positiveIntegrals(positiveIntegrals_),
                                                                                   negativeIntegrals(negativeIntegrals_),
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
    std::vector<cv::Mat> positiveImages, negativeImages;
    std::vector<Integrals> positivesIntegrals, negativesIntegrals;
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

        if ( !SampleExtractor::extractFromBigImage(positiveSamplesImage, positiveImages) )
        {
            std::cout << "Failed to load positive samples." << std::endl;
            return 6;
        }
        positivesIntegrals.resize(positiveImages.size());
        std::transform(positiveImages.begin(), positiveImages.end(),
                       positivesIntegrals.begin(),
                       ToIntegrals());
        std::cout << positivesIntegrals.size() << " positive samples loaded." << std::endl;

        if ( !SampleExtractor::extractFromBigImage(negativeSamplesImage, negativeSamplesIndex, negativeImages) )
        {
            std::cout << "Failed to load negative samples." << std::endl;
            return 7;
        }
        negativesIntegrals.resize(negativeImages.size());
        std::transform(negativeImages.begin(), negativeImages.end(),
                       negativesIntegrals.begin(),
                       ToIntegrals());
        std::cout << negativesIntegrals.size() << " negative samples loaded." << std::endl;
    }



    std::cout << "Optimizing Haar-like features..." << std::endl;

    tbb::concurrent_vector<ProbabilisticClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(wavelets, positivesIntegrals, negativesIntegrals, classifiers));



    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());

    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;

    //write all haar wavelets
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
