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



class ClassifierData : public MyHaarWavelet
{
protected:
    double stdDev;

public:
    ClassifierData() : MyHaarWavelet(),
                       stdDev(0) {}

    ClassifierData(const HaarWavelet & h)
    {
        rects.resize(h.dimensions());
        weights.resize(h.dimensions());
        for (unsigned int i = 0; i < h.dimensions(); ++i)
        {
            rects[i] = h.rect(i);
            weights[i] = h.weight(i);
        }
    }

    ClassifierData &operator=(const ClassifierData & c)
    {
        rects = c.rects;
        weights = c.weights;
        means = c.means;
        stdDev = c.stdDev;

        return *this;
    }

    void setMeans(const std::vector<double> & means_)
    {
        means.resize( means_.size() );
        for (unsigned int i = 0; i < means_.size(); ++i)
        {
            means[i] = means_[i];
        }
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

    bool operator < (const ClassifierData & rh) const
    {
        return stdDev < rh.stdDev;
    }
};



bool loadSamples(const boost::filesystem::path & samplesDir,
            std::vector<cv::Mat> & integralSums,
            std::vector<cv::Mat> & integralSquares)
{
    const boost::filesystem::directory_iterator end_iter;
    for( boost::filesystem::directory_iterator dir_iter(samplesDir) ; dir_iter != end_iter ; ++dir_iter)
    {
        if ( !boost::filesystem::is_regular_file(dir_iter->status()) )
        {
            std::cerr << dir_iter->path().native() << "is not a regular file." << std::endl;
            continue;
        }

        const std::string samplename = dir_iter->path().string();
        cv::Mat sample = cv::imread(samplename, CV_LOAD_IMAGE_GRAYSCALE);
        if (!sample.data)
        {
            std::cerr << "Failed to open file sample file " << samplename;
            continue;
        }

        cv::Mat integralSum(sample.rows + 1, sample.cols + 1, CV_64F);
        cv::Mat integralSquare(sample.rows + 1, sample.cols + 1, CV_64F);
        cv::integral(sample, integralSum, integralSquare, CV_64F);

        integralSums.push_back(integralSum);
        integralSquares.push_back(integralSquare);
    }

    return true;
}



void produceSrfs(mypca & pca, const HaarWavelet & wavelet, const std::vector<cv::Mat> & integralSums, const std::vector<cv::Mat> & integralSquares)
{
    const IntensityNormalizedWaveletEvaluator evaluator;

    const int records = integralSums.size();

    pca.set_num_variables(wavelet.dimensions());

    std::vector<double> srfsVector( wavelet.dimensions() );
    for (int i = 0; i < records; ++i)
    {
        evaluator.srfs(wavelet, integralSums[i], srfsVector);

        pca.add_record(srfsVector);
    }
}



/**
 * Returns the principal component with the smallest variance.
 */
void getOptimals(mypca & pca, ClassifierData & c)
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



void printSolution(mypca &pca)
{
    for(int i = 0; i < pca.get_num_variables(); ++i)
    {
        pca.get_eigenvector(i);
        std::cout << pca.get_eigenvalue(i) << " : (";

        bool first = true;
        const std::vector<double> eigenvector = pca.get_eigenvector(i);
        for (std::vector<double>::const_iterator it = eigenvector.begin(); it != eigenvector.end(); ++it)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                std::cout << ", ";
            }

            std::cout << *it;
        }

        std::cout << ")" << std::endl;
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
    std::vector<HaarWavelet> * wavelets;
    std::vector<cv::Mat> * integralSums;
    std::vector<cv::Mat> * integralSquares;
    tbb::concurrent_vector<ClassifierData> * classifiers;

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ClassifierData classifier( (*wavelets)[i] );

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
             tbb::concurrent_vector<ClassifierData> * classifiers_) : wavelets(wavelets_),
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



    tbb::concurrent_vector<ClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &integralSums, &integralSquares, &classifiers));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());


    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;


    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
