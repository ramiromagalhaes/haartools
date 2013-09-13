#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <armadillo>

#include "../libpca-1.2.11/include/pca.h"

#include "../ecrsgen/lib/haarwavelet.h"
#include "../ecrsgen/lib/haarwaveletutilities.h"

#include <tbb/tbb.h>



#define SAMPLE_SIZE 20



class mypca : public stats::pca //99% copy and paste
{
public:
    arma::Mat<double> cov_mat_;

    void solve() {
        assert_num_vars_();

        if (num_records_ < 2)
            throw std::logic_error("Number of records smaller than two.");

        data_.resize(num_records_, num_vars_);

        mean_ = stats::utils::compute_column_means(data_);
        stats::utils::remove_column_means(data_, mean_);

        sigma_ = stats::utils::compute_column_rms(data_);
        if (do_normalize_) stats::utils::normalize_by_column(data_, sigma_);

        arma::Col<double> eigval(num_vars_);
        arma::Mat<double> eigvec(num_vars_, num_vars_);

        cov_mat_ = stats::utils::make_covariance_matrix(data_);
        arma::eig_sym(eigval, eigvec, cov_mat_, solver_.c_str());
        arma::uvec indices = arma::sort_index(eigval, 1);

        for (long i=0; i<num_vars_; ++i) {
            eigval_(i) = eigval(indices(i));
            eigvec_.col(i) = eigvec.col(indices(i));
        }

        stats::utils::enforce_positive_sign_by_column(eigvec_);
        proj_eigvec_ = eigvec_;

        princomp_ = data_ * eigvec_;

        energy_(0) = arma::sum(eigval_);
        eigval_ *= 1./energy_(0);

        if (do_bootstrap_) bootstrap_eigenvalues_();
    }
};



struct ClassifierData
{
    HaarWavelet * wavelet;
    std::vector<double> mean;
    double stdDev;
    double q;

    ClassifierData() : wavelet(0), mean(0), stdDev(0), q(1) {}

    ClassifierData(const ClassifierData & c) : wavelet(c.wavelet),
                                               mean(c.mean),
                                               stdDev(c.stdDev),
                                               q(c.q) {}

    ClassifierData &operator=(const ClassifierData & c)
    {
        wavelet = c.wavelet;
        mean = c.mean;
        stdDev = c.stdDev;
        q = c.q;

        return *this;
    }

    bool write(std::ofstream & out) const
    {
        if ( !wavelet->write(out) )
        {
            return false;
        }

        for (unsigned int i = 0; i < mean.size(); i++)
        {
            out << ' ' << mean[i];
        }

        out << ' '
            << stdDev << ' '
            << q;

        return true;
    }
};



bool reverseClassifierDataSorter(const ClassifierData &c1, const ClassifierData &c2)
{
    return c1.stdDev < c2.stdDev;
}



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



void produceSrfs(mypca & pca, HaarWavelet * wavelet, std::vector<cv::Mat> & integralSums, std::vector<cv::Mat> & integralSquares)
{
    const int records = integralSums.size();

    pca.set_num_variables(wavelet->dimensions());

    std::vector<double> srfsVector( wavelet->dimensions() );
    for (int i = 0; i < records; ++i)
    {
        wavelet->setIntegralImages( &(integralSums[i]), &(integralSquares[i]) );
        wavelet->srfs( srfsVector );

        pca.add_record(srfsVector);
    }
}



/**
 * Returns the principal component with the smallest variance.
 */
void getOptimals(mypca & pca, ClassifierData & c)
{
    unsigned int min_value_index = 0;
    {
        double min_val = std::numeric_limits<double>::max();

        for (int i = 0; i < pca.get_num_variables(); ++i) //what's the smallest eigenvalue?
        {
            if ( pca.get_eigenvalue(i) < min_val )
            {
                min_value_index = i;
                min_val = pca.get_eigenvalue(i);
            }
        }

        std::vector<double> eigenvalues = pca.get_eigenvalues();
        std::min(eigenvalues.begin(), eigenvalues.end());
    }

    const std::vector<double> eigenvector = pca.get_eigenvector( min_value_index );

    for(unsigned int i = 0; i < eigenvector.size(); ++i)
    {
        c.wavelet->weight(i, (float) eigenvector[i]);
    }

    c.mean = pca.get_mean_values();

    c.stdDev = 0;
    std::vector<double> temp(eigenvector.size());
    for (unsigned int i = 0; i < eigenvector.size(); ++i)
    {
        temp[i] = 0;
        for (unsigned int j = 0; j < eigenvector.size(); ++j)
        {
            temp[i] += eigenvector[j] * pca.cov_mat_.at(j, i);
        }
    }
    for (unsigned int i = 0; i < eigenvector.size(); ++i)
    {
        c.stdDev += eigenvector[i] * temp[i];
    }

    c.stdDev = std::sqrt(c.stdDev);
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
 * Class (functor?) used by Intel TBB that will optimize the haar-like feature based classifiers
 * using PCA.
 */
class Optimize
{
    std::vector<HaarWavelet*> * wavelets;
    std::vector<cv::Mat> * integralSums;
    std::vector<cv::Mat> * integralSquares;
    tbb::concurrent_vector<ClassifierData> * classifiers;

public:
    void operator()(const tbb::blocked_range<std::vector<HaarWavelet*>::size_type> range) const
    {
        for(std::vector<HaarWavelet*>::size_type i = range.begin(); i != range.end(); ++i)
        {
            ClassifierData classifier;
            classifier.wavelet = (*wavelets)[i];

            mypca pca;
            produceSrfs(pca, classifier.wavelet, *integralSums, *integralSquares);
            pca.solve();

            getOptimals(pca, classifier);

            classifiers->push_back(classifier);
        }
    }

    Optimize(std::vector<HaarWavelet*> * wavelets_,
             std::vector<cv::Mat> * integralSums_,
             std::vector<cv::Mat> * integralSquares_,
             tbb::concurrent_vector<ClassifierData> * classifiers_) : wavelets(wavelets_),
                                                      integralSums(integralSums_),
                                                      integralSquares(integralSquares_),
                                                      classifiers(classifiers_) {}
};



/**
 *
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


    std::vector<HaarWavelet*> wavelets;
    std::vector<cv::Mat> integralSums, integralSquares;
    std::ofstream outputStream;


    {
        //Load a list of Haar wavelets
        std::cout << "Loading wavelets..." << std::endl;
        if (!loadHaarWavelets(&sampleSize, waveletsFileName, wavelets))
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



    std::cout << "Optimizing classifiers..." << std::endl;



    //TODO count progress
    tbb::concurrent_vector<ClassifierData> classifiers;
    tbb::parallel_for( tbb::blocked_range< std::vector<HaarWavelet*>::size_type >(0, wavelets.size()),
                       Optimize(&wavelets, &integralSums, &integralSquares, &classifiers));

    //sort the solutions using the variance. The smallest variance goes first
    tbb::parallel_sort(classifiers.begin(), classifiers.end(), reverseClassifierDataSorter);


    std::cout << "Done optimizing. Writing results to " <<  classifiersFileName << std::endl;


    //write all haar wavelets sorted from best to worst
    writeClassifiersData(outputStream, classifiers);

    return 0;
}
