#ifndef OPTIMIZATION_COMMONS_H
#define OPTIMIZATION_COMMONS_H

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include "mypca.h"

#include "haarwavelet.h"
#include "haarwaveletevaluators.h"



bool loadSamples(const boost::filesystem::path & samplesDir, std::vector<cv::Mat> & integralSums)
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
        cv::integral(sample, integralSum, CV_64F);

        integralSums.push_back(integralSum);
    }

    return true;
}



void produceSrfs(mypca & pca, const HaarWavelet & wavelet, const std::vector<cv::Mat> & integralSums)
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



#endif // OPTIMIZATION_COMMONS_H
