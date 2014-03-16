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



#define SAMPLE_SIZE 20



/**
 * Sets parameters to the weak classifier that creates a band over the SRFS,
 * as proposed in http://www.thinkmind.org/index.php?view=article&articleid=icons_2014_3_20_40057.
 * Data produced here can also be used as the PCA optimized Haar Wavelets.
 */
class BandClassifierData : public MyHaarWavelet
{
protected:
    double stdDev;

public:
    BandClassifierData() : MyHaarWavelet(),
                       stdDev(0) {}

    BandClassifierData(const HaarWavelet & h)
    {
        rects.resize(h.dimensions());
        weights.resize(h.dimensions());
        for (unsigned int i = 0; i < h.dimensions(); ++i)
        {
            rects[i] = h.rect(i);
            weights[i] = h.weight(i);
        }
    }

    BandClassifierData &operator=(const BandClassifierData & c)
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

    bool operator < (const BandClassifierData & rh) const
    {
        return stdDev < rh.stdDev;
    }
};



/**
 * Functor used by Intel TBB to optimize the haar-like feature based classifiers using PCA.
 */
class Optimize
{
    std::vector<HaarWavelet> * wavelets;
    std::vector<cv::Mat> * integralSums;
    tbb::concurrent_vector<BandClassifierData> * classifiers;

    /**
     * Returns the principal component with the smallest variance.
     */
    void getOptimals(mypca & pca, BandClassifierData & c) const
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

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet>::size_type> range) const
    {
        for(std::vector<HaarWavelet>::size_type i = range.begin(); i != range.end(); ++i)
        {
            BandClassifierData classifier( (*wavelets)[i] );

            mypca pca;
            produceSrfs(pca, &classifier, *integralSums);
            pca.solve();

            getOptimals(pca, classifier);

            classifiers->push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet> * wavelets_,
             std::vector<cv::Mat> * integralSums_,
             tbb::concurrent_vector<BandClassifierData> * classifiers_) : wavelets(wavelets_),
                                                                      integralSums(integralSums_),
                                                                      classifiers(classifiers_) {}
};



void writeClassifiersData(std::ofstream & outputStream, tbb::concurrent_vector<BandClassifierData> & classifiers)
{
    tbb::concurrent_vector<BandClassifierData>::const_iterator it = classifiers.begin();
    tbb::concurrent_vector<BandClassifierData>::const_iterator end = classifiers.end();
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
    if (argc != 4)
    {
        std::cout << "Usage " << argv[0] << " " << " WAVELETS_FILE SAMPLES_DIR OUTPUT_DIR" << std::endl;
        return 1;
    }

    const std::string waveletsFileName = argv[1];    //load Haar wavelets from here
    const std::string samplesFileName = argv[2];     //load samples from here
    const std::string classifiersFileName = argv[3]; //write output here



    std::vector<HaarWavelet> wavelets;
    std::vector<cv::Mat> integralSums;
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
        if ( !SampleExtractor::extractFromBigImage(samplesFileName, integralSums) )
        {
            std::cout << "Failed to load positive samples." << std::endl;
            return 6;
        }
        std::transform(integralSums.begin(), integralSums.end(),
                       integralSums.begin(),
                       ToIntegralSums());
        std::cout << integralSums.size() << " positive samples loaded." << std::endl;
    }



    std::cout << "Optimizing Haar-like features..." << std::endl;



    tbb::concurrent_vector<BandClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &integralSums, &classifiers));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end());


    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;


    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
