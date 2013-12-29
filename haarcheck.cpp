#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/unordered_set.hpp>

#include "haarwavelet.h"
#include "haarwaveletutilities.h"

#define SAMPLE_SIZE 20

#define MIN_RECT_HEIGHT 3 //Minimum = 3 thanks to Pavani's restriction #6.
#define MIN_RECT_WIDTH 3 //Minimum = 3 thanks to Pavani's restriction #6.



/*
 * Pavani's restrictions on Haar wavelets generation:
 * 1) only 2 to 4 rectangles
 * 2) detector size = 20x20
 * 3) no rotated rectangles
 * 4) disjoint rectangles are away of each other an integer multiple of rectangle sizes
 * 5) all rectangles in a HW have the same size
 * 6) no rectangles smaller than 3x3
 */



inline bool same(const cv::Rect &r1, const cv::Rect &r2) {
    return r1.x      == r2.x
        && r1.y      == r2.y
        && r1.width  == r2.width
        && r1.height == r2.height;
}



inline int contains(std::vector<cv::Rect>::const_iterator it, const std::vector<cv::Rect>::const_iterator end, const cv::Rect &r)
{
    int i = 0;

    for(; it != end; ++it)
    {
        if (same(*it, r))
        {
            i++;
        }
    }

    return i;
}



inline bool same(HaarWavelet & w1, HaarWavelet & w2)
{
    if (w1.dimensions() != w2.dimensions())
    {
        return false;
    }

    const std::vector<cv::Rect>::const_iterator begin1 = w1.rects_begin();
    const std::vector<cv::Rect>::const_iterator end1 = w1.rects_end();
    const std::vector<cv::Rect>::const_iterator end2 = w2.rects_end();
    std::vector<cv::Rect>::const_iterator it1 = w1.rects_begin();
    for(; it1 != end1; ++it1)
    {
        const int amountOfThisRectangle = contains(begin1, end1, *it1);

        std::vector<cv::Rect>::const_iterator it2 = w2.rects_begin();
        if ( contains(it2, end2, *it1) != amountOfThisRectangle)
        {
            return false;
        }
    }

    return true;
}



inline bool hasOverlappingRectangles(const HaarWavelet & w1)
{
    std::vector<cv::Rect>::const_iterator it1 = w1.rects_begin();
    const std::vector<cv::Rect>::const_iterator end1 = w1.rects_end();
    for(; it1 != end1; ++it1)
    {
        std::vector<cv::Rect>::const_iterator it2 = w1.rects_begin();
        const std::vector<cv::Rect>::const_iterator end2 = w1.rects_end();
        for(; it2 != end2; ++it2)
        {
            if (it1 != it2 && same(*it1, *it2))
            {
                return true;
            }
        }
    }

    return false;
}



/**
 * Checks if the Haar-like features generated by haargen conform to Pavani's restrictions.
 */
int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    cv::Size sampleSize(SAMPLE_SIZE, SAMPLE_SIZE); //size in pixels of the trainning images


    //load the wavelets
    std::vector<HaarWavelet> wavelets;
    std::cout << "Loading Haar wavelets from " << args[1] << std::endl;
    loadHaarWavelets(args[1], wavelets);
    std::cout << "Loaded " << wavelets.size() << " wavelets." << std::endl;

    //STATS
    int dimensions[3] = {0, 0, 0}; //2, 3 and 4 dimensions of the wavelets
    int regionsHistogram[3][3]; //x regions are 8 - 4 - 8
                                //y regions are 7 - 6 - 7
    int widthHistogram[20], heightHistogram[20];
    int totalRectangles;
    {
        for(int i = 0; i < 20; ++i)
        {
            widthHistogram[i] = heightHistogram[i] = 0;
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                regionsHistogram[i][j] = 0;
            }
        }


        std::vector<HaarWavelet>::iterator it = wavelets.begin();
        const std::vector<HaarWavelet>::iterator end = wavelets.end();
        for(;it != end; ++it)
        {
            dimensions[it->dimensions() - 2]++;

            std::vector<cv::Rect>::const_iterator itr = it->rects_begin();
            const std::vector<cv::Rect>::const_iterator endr = it->rects_end();
            for(; itr != endr; ++itr)
            {
                const cv::Rect r = *itr;
                totalRectangles++;
                widthHistogram[r.width - 1]++;
                heightHistogram[r.height - 1]++;

                float horizontalMean = r.x + r.width / 2.0f;
                float verticalMean = r.y + r.height / 2.0f;

                const int xIndex = horizontalMean < 8 ? 0 :
                              8 <= horizontalMean && horizontalMean < 12 ? 1 :
                                                                           2;
                const int yIndex = verticalMean < 7 ? 0 :
                              7 <= verticalMean && verticalMean < 13 ? 1 :
                                                                       2;
                regionsHistogram[xIndex][yIndex]++;
            }
        }

        std::cout << "Total 2D/3D/4D wavelets: " << dimensions[0] << "/" << dimensions[1] << "/" << dimensions[2] << std::endl;
        std::cout << "Total rectangles: " << totalRectangles << std::endl;

        std::stringstream wHist, hHist;
        wHist << "Width histogram: ";
        hHist << "Height histogram: ";
        for(int i = 0; i < 20; ++i)
        {
            wHist << widthHistogram[i] << " ";
            hHist << heightHistogram[i] << " ";
        }
        std::cout << wHist.str() << std::endl;
        std::cout << hHist.str() << std::endl;

        std::cout << "Rectangles mean position 2D histogram: " << std::endl;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << regionsHistogram[j][i] << " ";
            }
            std::cout << std::endl;
        }
    }



    {//double checks for wavelets with overlapping rectangles
       std::cout << "Checking for overlapped rectangles..." << std::endl;
       std::vector<HaarWavelet>::iterator it = wavelets.begin();
       const std::vector<HaarWavelet>::iterator end = wavelets.end();
       for(;it != end; ++it)
       {
           if (hasOverlappingRectangles(*it))
           {
               std::cout << "Overlaps ==> ";
               it->write(std::cout);
               std::cout << std::endl;
           }
       }
    }



    {//double checks if any rect has x or y at position 20 or more
        std::cout << "Checking for problems with rectangle sizes..." << std::endl;
        std::vector<HaarWavelet>::iterator it = wavelets.begin();
        const std::vector<HaarWavelet>::iterator end = wavelets.end();
        for(;it != end; ++it)
        {
            std::vector<cv::Rect>::const_iterator itr = it->rects_begin();
            const std::vector<cv::Rect>::const_iterator endr = it->rects_end();
            for(; itr != endr; ++itr)
            {
                if (itr->x >= 20 || itr->y >= 20 || itr->x < 0 || itr->y < 0 || itr->x + itr->width > 20 || itr->y + itr->height > 20 || itr->width < 3 || itr->height < 3)
                {
                    std::cout << "Size problem ==> ";
                    it->write(std::cout);
                    std::cout << std::endl;
                    break;
                }
            }
        }
    }



    {//double checks for repeated rects in a haar wavelet through brute force
        std::cout << "Checking duplicated rects in each haar wavelet..." << std::endl;
        std::vector<HaarWavelet>::iterator it = wavelets.begin();
        const std::vector<HaarWavelet>::iterator end = wavelets.end();
        for(;it != end; ++it)
        {
            std::vector<HaarWavelet>::iterator it2 = it + 1;
            for(;it2 != end; ++it2)
            {
                if( same(*it, *it2) )
                {
                    std::cout << "Repeats ==> ";
                    it->write(std::cout);
                    std::cout << std::endl;
                }
            }
        }
    }

    return 0;
}
