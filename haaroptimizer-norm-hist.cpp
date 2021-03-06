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

#include "optimization_commons.h"
#include "mypca.h"

#include "haarwavelet.h"
#include "haarwaveletutilities.h"
#include "haarwaveletevaluators.h"

#include "sampleextractor.h"

#include <tbb/tbb.h>



#define HISTOGRAM_BUCKETS 128



/**
 * Used to write data describing the positive instances as a normal distribution and
 * the negative instances as a histogram. The positive and negative instances have their
 * own weight.
 */
class ProbabilisticClassifierData : public DualWeightHaarWavelet
{
public:
    ProbabilisticClassifierData() : mean(.0),
                                    stdDev(1.0) {}

    ProbabilisticClassifierData(const HaarWavelet & wavelet) : mean(.0),
                                                               stdDev(1.0)
    {
        for(std::vector<cv::Rect>::const_iterator it = wavelet.rects_begin(); it != wavelet.rects_end(); ++it)
        {
            rects.push_back(*it);
        }

        for(std::vector<float>::const_iterator it = wavelet.weights_begin(); it != wavelet.weights_end(); ++it)
        {
            weightsPositive.push_back(*it);
            weightsNegative.push_back(*it);
        }
    }

    ProbabilisticClassifierData(const DualWeightHaarWavelet & wavelet) : DualWeightHaarWavelet(wavelet),
                                                                         mean(.0),
                                                                         stdDev(1.0) {}

    ProbabilisticClassifierData& operator=(const ProbabilisticClassifierData & c)
    {
        rects = c.rects;
        weightsPositive = c.weightsPositive;
        weightsNegative = c.weightsNegative;
        mean = c.mean;
        stdDev = c.stdDev;
        histogram = c.histogram;

        return *this;
    }

    void setPositiveWeights(const std::vector<double> & projection_)
    {
        DualWeightHaarWavelet::weightsPositive.reserve(dimensions());

        for(unsigned int i = 0; i < projection_.size(); ++i)
        {
            DualWeightHaarWavelet::weightsPositive[i] = projection_[i];
        }
    }

    void setNegativeWeights(const std::vector<double> & projection_)
    {
        DualWeightHaarWavelet::weightsNegative.reserve(dimensions());

        for(unsigned int i = 0; i < projection_.size(); ++i)
        {
            DualWeightHaarWavelet::weightsNegative[i] = projection_[i];
        }
    }

    void setStdDev(const double stdDev_)
    {
        stdDev = stdDev_;
    }

    void setMean(const double mean_)
    {
        mean = mean_;
    }

    void setHistogram(std::vector<double> histogram_)
    {
        histogram = histogram_;
    }

    void setPositivePrior(double p)
    {
        positivePrior = p;
    }

    void setNegativePrior(double p)
    {
        negativePrior = p;
    }

    bool operator < (const ProbabilisticClassifierData & rh) const
    {
        return stdDev < rh.stdDev;
    }

    bool write(std::ostream &output) const
    {
        if ( !DualWeightHaarWavelet::write(output) )
        {
            return false;
        }

        output << ' '
               << positivePrior << ' '
               << mean << ' '
               << stdDev << ' '
               << negativePrior << ' '
               << histogram.size();

        for (unsigned int i = 0; i < histogram.size(); ++i)
        {
            output << ' ' << histogram[i];
        }

        return true;
    }

private:
    double mean; //statistics taken from the feature value, not directly from the SRFS
    double stdDev;

    double positivePrior, negativePrior;

    std::vector<double> histogram;
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

    void getOptimalsForPositiveSamples(mypca & pca, ProbabilisticClassifierData & c) const
    {
        //The highest variance eigenvector is the first one.
        //c.setPositiveWeights(pca.get_eigenvector(0));

        {
            //This block sets the mean. It is acquired from the projection of the
            //mean values of the PCA in the direction of the eigenvector (weights).
            std::vector<double> meanSrfs = pca.get_mean_values();
            const double mean = std::inner_product(c.weightsPositive_begin(),
                                                   c.weightsPositive_end(),
                                                   meanSrfs.begin(), .0);
            c.setMean( mean );
        }

        {
            //This block sets the standard deviation. It is acquired from the projection
            //of the covariance matrix in the directin of the eigenvector (weights).
            std::vector<double> temp(c.dimensions());
            for (unsigned int i = 0; i < c.dimensions(); ++i)
            {
                std::vector<double> column = stats::utils::extract_column_vector(pca.cov_mat_, i);
                temp[i] = std::inner_product(c.weightsPositive_begin(),
                                             c.weightsPositive_end(),
                                             column.begin(), .0);
            }
            const double stdDev = std::sqrt( std::inner_product(c.weightsPositive_begin(),
                                                                c.weightsPositive_end(),
                                                                temp.begin(), .0) );
            c.setStdDev(stdDev);
        }
    }



    void getOptimalsForNegativeSamples(mypca & pca, ProbabilisticClassifierData & c) const
    {
        //c.setNegativeWeights(pca.get_eigenvector(0));

        std::vector<double> histogram(HISTOGRAM_BUCKETS);
        std::fill(histogram.begin(), histogram.end(), .0);

        const double increment = 1.0/pca.get_num_records();

        for (long i = 0; i < pca.get_num_records(); ++i)
        {
            std::vector<double> r = pca.get_record(i);
            const double featureValue = std::inner_product(c.weightsNegative_begin(),
                                                           c.weightsNegative_end(),
                                                           r.begin(), .0);

            //increment bin count
            const int index = featureValue >= std::sqrt(2) ? histogram.size() :
                              featureValue <= -std::sqrt(2) ? 0 :
                              (int)((histogram.size()/2.0) * featureValue / std::sqrt(2)) + histogram.size()/2;
            histogram[index] += increment;
        }

        c.setHistogram(histogram);
    }

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ProbabilisticClassifierData classifier( (*wavelets)[i] );

            const double positivePrior = double(positivesIntegralSums->size()) / (positivesIntegralSums->size() + negativesIntegralSums->size());
            classifier.setPositivePrior(positivePrior);
            classifier.setNegativePrior(1.0 - positivePrior);

            {
                mypca positive_samples_pca;
                produceSrfs(positive_samples_pca, &classifier, *positivesIntegralSums);
                positive_samples_pca.solve();
                getOptimalsForPositiveSamples(positive_samples_pca, classifier);
            }

            {
                mypca negative_samples_pca;
                produceSrfs(negative_samples_pca, &classifier, *negativesIntegralSums);
                negative_samples_pca.solve();
                getOptimalsForNegativeSamples(negative_samples_pca, classifier);
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

    const std::string waveletsFileName = argv[1];     //load Haar wavelets from here
    const std::string positiveSamplesImage = argv[2]; //load + samples from here
    const std::string negativeSamplesImage = argv[3]; //load - samples from here
    const std::string negativeSamplesIndex = argv[4]; //load - samples from here
    const std::string classifiersFileName = argv[5];  //write output here



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
