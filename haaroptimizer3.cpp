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



class ProbabilisticClassifierData : public DualWeightHaarWavelet
{
public:
    ProbabilisticClassifierData() : positiveMean(.0), positiveStdDev(1.0),
                                    negativeMean(.0), negativeStdDev(1.0) {}

    ProbabilisticClassifierData(const HaarWavelet & wavelet) : ProbabilisticClassifierData()
    {
        std::vector<cv::Rect>::const_iterator it = wavelet.rects_begin();

        for(; it != wavelet.rects_end(); ++it)
        {
            rects.push_back(*it);
        }

        weightsPositive.resize(wavelet.dimensions(), 0);
        weightsNegative.resize(wavelet.dimensions(), 0);
    }

    ProbabilisticClassifierData(const DualWeightHaarWavelet & wavelet) : DualWeightHaarWavelet(wavelet),
                                                                         positiveMean(.0), positiveStdDev(1.0),
                                                                         negativeMean(.0), negativeStdDev(1.0) {}

    ProbabilisticClassifierData& operator=(const ProbabilisticClassifierData & c)
    {
        rects = c.rects;
        weightsPositive = c.weightsPositive;
        weightsNegative = c.weightsNegative;

        positiveMean = c.positiveMean;
        positiveStdDev = c.positiveStdDev;
        negativeMean = c.negativeMean;
        negativeStdDev = c.negativeStdDev;

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

    void setPositiveStdDev(const double stdDev_)
    {
        positiveStdDev = stdDev_;
    }

    void setPositiveMean(const double mean_)
    {
        positiveMean = mean_;
    }

    void setNegativeStdDev(const double stdDev_)
    {
        negativeStdDev = stdDev_;
    }

    void setNegativeMean(const double mean_)
    {
        negativeMean = mean_;
    }

    bool operator < (const ProbabilisticClassifierData & rh) const
    {
        return positiveStdDev < rh.positiveStdDev;
    }

    bool write(std::ostream &output) const
    {
        if ( !DualWeightHaarWavelet::write(output) )
        {
            return false;
        }

        output << ' '
               << positiveMean << ' '
               << positiveStdDev << ' '
               << negativeMean << ' '
               << negativeStdDev;

        return true;
    }

private:
     //statistics taken from the feature value, not directly from the SRFS
    double positiveStdDev;
    double positiveMean;
    double negativeStdDev;
    double negativeMean;
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
        c.setPositiveWeights(pca.get_eigenvector(0));

        {
            //This block sets the mean. It is acquired from the projection of the
            //mean values of the PCA in the direction of the eigenvector (weights).
            std::vector<double> meanSrfs = pca.get_mean_values();
            const double mean = std::inner_product(c.weightsPositive_begin(),
                                                   c.weightsPositive_end(),
                                                   meanSrfs.begin(), .0);
            c.setPositiveMean( mean );
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
            c.setPositiveStdDev(stdDev);
        }
    }



    void getOptimalsForNegativeSamples(mypca & pca, ProbabilisticClassifierData & c) const
    {
        //The highest variance eigenvector is the first one.
        c.setNegativeWeights(pca.get_eigenvector(0));

        {
            //This block sets the mean. It is acquired from the projection of the
            //mean values of the PCA in the direction of the eigenvector (weights).
            std::vector<double> meanSrfs = pca.get_mean_values();
            const double mean = std::inner_product(c.weightsPositive_begin(),
                                                   c.weightsPositive_end(),
                                                   meanSrfs.begin(), .0);
            c.setNegativeMean( mean );
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
            c.setNegativeStdDev(stdDev);
        }
    }



public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ProbabilisticClassifierData classifier( (*wavelets)[i] );

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
//    Optimize opt(&wavelets, &positivesIntegralSums, &negativesIntegralSums, &classifiers);
//    opt(tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());

    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;

    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
