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

#include "mypca.h"

#include "haarwavelet.h"
#include "haarwaveletutilities.h"
#include "haarwaveletevaluators.h"

#include <tbb/tbb.h>



#define SAMPLE_SIZE 20



class ClassifierData : public HaarWavelet
{
    void setWeights(const std::vector<double> & weights_)
    {
        weights.reserve(weights_.size());
        for (unsigned int i = 0; i < weights_.size(); ++i)
        {
            weights[i] = weights_[i];
        }
    }
};


class GaussianClassifierData : public ClassifierData
{
private:
    double stdDev; //statistics taken from the feature value, not directly from the SRFS
    double mean;

public:
    void setStdDev(const double stdDev_)
    {
        stdDev = stdDev_;
    }

    void setMean(const double mean_)
    {
        mean = mean_;
    }
};



class HistogramClassifierData : public ClassifierData
{
private:
    std::vector<double> histogram;
public:
    HistogramClassifierData() : histogram(100) {}

    void setHistogram(std::vector<double> histogram_)
    {
        histogram = histogram_;
    }
};



/**
 * Returns the principal component with the smallest variance.
 */
void getOptimals(mypca & pca, GaussianClassifierData & c)
{
    //The smallest eigenvalue is the last one
    const std::vector<double> eigenvector = pca.get_eigenvector( pca.get_num_variables() - 1 );

    c.setWeights(eigenvector);
    c.setMeans( pca.get_mean_values() );

    double stdDev = 0;
    std::vector<double> temp(eigenvector.size());
    for (unsigned int i = 0; i < eigenvector.size(); ++i)
    {
        std::vector<double> column = stats::utils::extract_column_vector(pca.cov_mat_, i);
        temp[i] = std::inner_product(eigenvector.begin(), eigenvector.end(),
                                     column.begin(), .0);
    }
    stdDev = std::sqrt( std::inner_product(eigenvector.begin(), eigenvector.end(), temp.begin(), .0) );

    c.setStdDev(stdDev);
}



void writeClassifiersData(std::ofstream & outputStream, tbb::concurrent_vector<GaussianClassifierData> & classifiers)
{
    tbb::concurrent_vector<GaussianClassifierData>::const_iterator it = classifiers.begin();
    tbb::concurrent_vector<GaussianClassifierData>::const_iterator end = classifiers.end();
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
    std::vector<cv::Mat> * integralSums;
    std::vector<cv::Mat> * integralSquares;
    tbb::concurrent_vector<GaussianClassifierData> * classifiers;

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            GaussianClassifierData classifier( (*wavelets)[i] );

            mypca pca;
            produceSrfs(pca, classifier, *integralSums, *integralSquares);
            pca.solve();

            getOptimals(pca, classifier);

            classifiers->push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet> * wavelets_,
             std::vector<cv::Mat> * integralSums_,
             std::vector<cv::Mat> * integralSquares_,
             tbb::concurrent_vector<GaussianClassifierData> * classifiers_) : wavelets(wavelets_),
                                                      integralSums(integralSums_),
                                                      integralSquares(integralSquares_),
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
        std::cout << "Usage " << argv[0] << " " << " WAVELETS_FILE SAMPLES_DIR OUTPUT_DIR" << std::endl;
        return 1;
    }

    const std::string waveletsFileName = argv[1];    //load Haar wavelets from here
    const std::string samplesDirName = argv[2];      //load samples from here
    const std::string classifiersFileName = argv[3]; //write output here

    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images


    std::vector<HaarWavelet> wavelets;
    std::vector<cv::Mat> integralSums, integralSquares;
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

        //Check if the samples directory exist and is a directory
        const boost::filesystem::path samplesDir(samplesDirName);
        if ( !boost::filesystem::exists(samplesDirName) || !boost::filesystem::is_directory(samplesDirName) )
        {
            std::cout << "Sample directory " << samplesDir << " does not exist or is not a directory." << std::endl;
            return 3;
        }

        outputStream.open(classifiersFileName.c_str(), std::ios::trunc);
        if ( !outputStream.is_open() )
        {
            std::cout << "Can't open output file." << std::endl;
            return 5;
        }

        std::cout << "Loading samples..." << std::endl;
        if ( !loadSamples(samplesDir, integralSums, integralSquares) )
        {
            std::cout << "Failed to load samples." << std::endl;
            return 6;
        }
        std::cout << integralSums.size() << " samples loaded." << std::endl;
    }



    std::cout << "Optimizing Haar-like features..." << std::endl;



    tbb::concurrent_vector<GaussianClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &integralSums, &integralSquares, &classifiers));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());


    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;


    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
