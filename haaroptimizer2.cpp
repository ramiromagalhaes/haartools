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

#include <tbb/tbb.h>



#define SAMPLE_SIZE 20



class ClassifierData : public HaarWavelet
{
public:
    ClassifierData() : histogram(100) {}

    ClassifierData(const HaarWavelet & wavelet) : histogram(100)
    {
        rects.resize(wavelet.dimensions());
        weights.resize(wavelet.dimensions());
        for (unsigned int i = 0; i < wavelet.dimensions(); ++i)
        {
            rects[i] = wavelet.rect(i);
            weights[i] = wavelet.weight(i);
        }
    }

    ClassifierData& operator=(const ClassifierData & c)
    {
        rects = c.rects;
        weights = c.weights;
        mean = c.mean;
        stdDev = c.stdDev;
        histogram = c.histogram;

        return *this;
    }

    void setWeights(const std::vector<double> & weights_)
    {
        weights.reserve(weights_.size());
        for (unsigned int i = 0; i < weights_.size(); ++i)
        {
            weights[i] = weights_[i];
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

    bool operator < (const ClassifierData & rh) const
    {
        return stdDev < rh.stdDev;
    }

private:
    double stdDev; //statistics taken from the feature value, not directly from the SRFS
    double mean;
    std::vector<double> histogram;
};



void getOptimalsForPositiveSamples(mypca & pca, ClassifierData & c)
{
    {
        //This block sets the mean. It is acquired from the projection of the
        //mean values of the PCA in the direction of the eigenvector (weights).
        std::vector<double> meanSrfs = pca.get_mean_values();
        const double mean = std::inner_product(c.weights_begin(),
                                               c.weights_end(),
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
            temp[i] = std::inner_product(c.weights_begin(),
                                         c.weights_end(),
                                         column.begin(), .0);
        }
        const double stdDev = std::sqrt( std::inner_product(c.weights_begin(),
                                                            c.weights_end(),
                                                            temp.begin(), .0) );
        c.setStdDev(stdDev);
    }
}



void getOptimalsForNegativeSamples(mypca & pca, ClassifierData & c)
{
    std::vector<double> histogram(100);
    const double increment = 1.0/pca.get_num_records();

    for (long i = 0; i < pca.get_num_records(); ++i)
    {
        std::vector<double> r = pca.get_record(i);
        double featureValue = std::inner_product(c.weights_begin(),
                                                 c.weights_end(),
                                                 r.begin(), .0);

        //increment bin count
        int index = featureValue >= std::sqrt(2) ? 100 :
                    featureValue <= -std::sqrt(2) ? 0 :
                    (int)(50 * featureValue / std::sqrt(2)) + 50;
        histogram[index] += increment;
    }
}



void writeClassifiersData(std::ofstream & outputStream, tbb::concurrent_vector<ClassifierData> & classifiers)
{
    tbb::concurrent_vector<ClassifierData>::const_iterator it = classifiers.begin();
    tbb::concurrent_vector<ClassifierData>::const_iterator end = classifiers.end();
    for(; it != end; ++it)
    {
        it->write(outputStream);
        outputStream << '\n';
    }
}



/**
 * Functor used by Intel TBB to optimize the haar-like feature based classifiers using PCA.
 */
class Optimize
{
private:
    std::vector<HaarWavelet> * wavelets;
    std::vector<cv::Mat> * positivesIntegralSums;
    std::vector<cv::Mat> * negativesIntegralSums;
    tbb::concurrent_vector<ClassifierData> * classifiers;

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ClassifierData classifier( (*wavelets)[i] );

            {
                mypca positive_samples_pca;
                produceSrfs(positive_samples_pca, classifier, *positivesIntegralSums);
                positive_samples_pca.solve();
                getOptimalsForPositiveSamples(positive_samples_pca, classifier);
            }

            {
                mypca negative_samples_pca;
                produceSrfs(negative_samples_pca, classifier, *negativesIntegralSums);
                negative_samples_pca.solve();
                getOptimalsForNegativeSamples(negative_samples_pca, classifier);
            }

            classifiers->push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet> * wavelets_,
             std::vector<cv::Mat> * positivesIntegralSums_,
             std::vector<cv::Mat> * negativesIntegralSums_,
             tbb::concurrent_vector<ClassifierData> * classifiers_) : wavelets(wavelets_),
                                                                      positivesIntegralSums(positivesIntegralSums_),
                                                                      negativesIntegralSums(negativesIntegralSums_),
                                                                      classifiers(classifiers_) {}

};



/**
 * Loads the Haar wavelets from a file and the image samples found in a directory, then produce
 * the SRFS for each Haar wavelet. Extract the principal component of least variance and use it
 * as the new weights of the respective Haar wavelet. When all is done, write the 'optimized'
 * Haar wavelets to a file.
 */
int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage " << argv[0] << " " << " WAVELETS_FILE POSITIVE_SAMPLES_DIR NEGATIVE_SAMPLES_DIR OUTPUT_DIR" << std::endl;
        return 1;
    }

    const std::string waveletsFileName = argv[1];       //load Haar wavelets from here
    const std::string positiveSamplesDirName = argv[2]; //load + samples from here
    const std::string negativeSamplesDirName = argv[3]; //load - samples from here
    const std::string classifiersFileName = argv[4];    //write output here



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

        //Check if the positive samples directory exist and is a directory
        const boost::filesystem::path positiveSamplesDir(positiveSamplesDirName);
        if ( !boost::filesystem::exists(positiveSamplesDir) || !boost::filesystem::is_directory(positiveSamplesDir) )
        {
            std::cout << "Sample directory " << positiveSamplesDir << " does not exist or is not a directory." << std::endl;
            return 3;
        }

        //Check if the negative samples directory exist and is a directory
        const boost::filesystem::path negativeSamplesDir(negativeSamplesDirName);
        if ( !boost::filesystem::exists(negativeSamplesDir) || !boost::filesystem::is_directory(negativeSamplesDir) )
        {
            std::cout << "Sample directory " << negativeSamplesDir << " does not exist or is not a directory." << std::endl;
            return 4;
        }

        outputStream.open(classifiersFileName.c_str(), std::ios::trunc);
        if ( !outputStream.is_open() )
        {
            std::cout << "Can't open output file." << std::endl;
            return 5;
        }

        std::cout << "Loading positive samples..." << std::endl;
        if ( !loadSamples(positiveSamplesDir, positivesIntegralSums) )
        {
            std::cout << "Failed to load positive samples." << std::endl;
            return 6;
        }
        std::cout << positivesIntegralSums.size() << " samples loaded." << std::endl;

        std::cout << "Loading negative samples..." << std::endl;
        if ( !loadSamples(negativeSamplesDir, negativesIntegralSums) )
        {
            std::cout << "Failed to load negative samples." << std::endl;
            return 7;
        }
        std::cout << negativesIntegralSums.size() << " samples loaded." << std::endl;
    }



    std::cout << "Optimizing Haar-like features..." << std::endl;



    tbb::concurrent_vector<ClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &positivesIntegralSums, &negativesIntegralSums, &classifiers));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());

    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;

    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
