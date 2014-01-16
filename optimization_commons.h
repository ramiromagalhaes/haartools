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



struct ToIntegralSums
{
    inline cv::Mat operator()(cv::Mat & image) const
    {
        cv::Mat iSum(image.rows, image.cols, cv::DataType<double>::type);
        cv::integral(image, iSum, cv::DataType<double>::type);
        return iSum;
    }
};



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



#endif // OPTIMIZATION_COMMONS_H
