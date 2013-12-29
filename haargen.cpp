#include <string>
#include <iostream>
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



bool same(const cv::Rect &r1, const cv::Rect &r2) {
    return r1.x      == r2.x
        && r1.y      == r2.y
        && r1.width  == r2.width
        && r1.height == r2.height;
}



int contains(std::vector<cv::Rect>::const_iterator it, const std::vector<cv::Rect>::const_iterator end, const cv::Rect &r)
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



struct wavelet_equals : std::binary_function<const HaarWavelet &, const HaarWavelet &, bool>
{
    /**
     * 2 Haar wavelets will be the same if their dimensions and rects are the same. The
     * problem is that the list of rectangles are unordered and there might be repeated
     * rectangles inside both wavelets (there should not be, though).
     */
    bool operator()(const HaarWavelet & w1, const HaarWavelet & w2) const
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
};



struct wavelet_hash : std::unary_function<const HaarWavelet &, std::size_t>
{
    std::size_t operator()(const HaarWavelet & w) const
    {
        std::size_t hashval = 0;

        std::vector<cv::Rect>::const_iterator it = w.rects_begin();
        const std::vector<cv::Rect>::const_iterator end = w.rects_end();
        for(; it != end; ++it)
        {
            hashval += it->x * it->y * it->width * it->height;
        }

        hashval += 160000 * (w.dimensions() - 2);
        return hashval;
    }
};



typedef boost::unordered_set<HaarWavelet, wavelet_hash, wavelet_equals> WaveletMap;



struct wavelet_comparator {
    bool operator()(const HaarWavelet & w1, const HaarWavelet & w2) const
    {
        if (w1.dimensions() != w2.dimensions())
        {
            return w1.dimensions() < w2.dimensions();
        }

        return wavelet_hash()(w1) < wavelet_hash()(w2);
    }
};



/**
 * Generates Haar wavelets with 2 rectangles.
 */
void gen2d(WaveletMap &wavelets)
{
    std::vector<float> weights(2);
    weights[0] = 1;
    weights[1] = -1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w++)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h++)
        {
            for(int x = 0; x <= SAMPLE_SIZE - w; x+=2) //x position of the first rectangle
            {
                for(int y = 0; y <= SAMPLE_SIZE - h; y+=2) //y position of the first rectangle
                {
                    if (   x + w > SAMPLE_SIZE
                        || y + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    for(int dx = -SAMPLE_SIZE / w; dx < SAMPLE_SIZE / w; dx++) //dx = horizontal displacement multiplier of the second rectangle.
                    {                                           //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                        for(int dy = -SAMPLE_SIZE / h; dy < SAMPLE_SIZE / h; dy++) //dy is similar to dx but in the vertical direction
                        {
                            if (dx == 0 && dy == 0) //rectangles will overlap
                            {
                                continue;
                            }

                            const int xOther = x + dx * w;
                            const int yOther = y + dy * h;

                            if (   xOther < 0
                                || yOther < 0
                                || xOther >= SAMPLE_SIZE
                                || yOther >= SAMPLE_SIZE
                                || xOther + w > SAMPLE_SIZE
                                || yOther + h > SAMPLE_SIZE)
                            {
                                continue;
                            }

                            //create the wavelet
                            std::vector<cv::Rect> rects(2);
                            rects[0] = cv::Rect(     x,      y, w, h);
                            rects[1] = cv::Rect(xOther, yOther, w, h);

                            HaarWavelet wavelet(rects, weights);
                            wavelets.insert( wavelet );
                        }
                    }
                }
            }
        }
    }
}



/**
 * Generates Haar wavelets with 3 rectangles.
 */
void gen3d(WaveletMap &wavelets)
{
    const int K = 3; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w+=2)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h+=2)
        {
            int x[K], //x and y positions of each rectangle.
                y[K];

            for(x[0] = 0; x[0] <= SAMPLE_SIZE - w; x[0]+=2) //for each x...
            {
                for(y[0] = 0; y[0] <= SAMPLE_SIZE - h; y[0]+=2) //...and y of the first rectangle...
                {
                    if (   x[0] + w > SAMPLE_SIZE
                        || y[0] + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    int dx[K - 1], //dx = horizontal displacement multiplier of the second rectangle.
                        dy[K - 1]; //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                                   //dy is similar to dx but in the vertical direction

                    for(dx[0] = -SAMPLE_SIZE / w; dx[0] < SAMPLE_SIZE / w; dx[0]++)
                    {
                        for(dy[0] = -SAMPLE_SIZE / h; dy[0] < SAMPLE_SIZE / h; dy[0]++)
                        {
                            for(dx[1] = -SAMPLE_SIZE / w; dx[1] < SAMPLE_SIZE / w; dx[1]++)
                            {
                                for(dy[1] = -SAMPLE_SIZE / h; dy[1] < SAMPLE_SIZE / h; dy[1]++)
                                {
                                    //avoids rectangle overlapping
                                    if (   (dx[0] == 0 && dy[0] == 0)
                                        || (dx[1] == 0 && dy[1] == 0))
                                    {
                                        continue;
                                    }

                                    //sets the values of the x, y position of the rectangles
                                    for (int i = 1; i < K; i++)
                                    {
                                        x[i] = x[i-1] + dx[i-1] * w;
                                        y[i] = y[i-1] + dy[i-1] * h;
                                    }

                                    {//avoids rectangles overlapping
                                        bool overlaps = false;

                                        for (int i = 0; i < K; i++)
                                        {
                                            for (int j = 0; j < K; j++)
                                            {
                                                if (i != j && x[i] == x[j] && y[i] == y[j])
                                                {
                                                    overlaps = true;
                                                    break;
                                                }
                                            }
                                            if (overlaps)
                                            {
                                                break;
                                            }
                                        }
                                        if (overlaps)
                                        {
                                            continue;
                                        }
                                    }

                                    {
                                        bool overflow = false;
                                        for (int i = 1; i < K; i++) //...and all rectangles fit into the sampling window...
                                        {
                                            if (   x[i] < 0
                                                || y[i] < 0
                                                || x[i] >= SAMPLE_SIZE //x and y must be at least 1 pixel away from the window's last pixel
                                                || y[i] >= SAMPLE_SIZE
                                                || x[i] + w > SAMPLE_SIZE //and the rectangle must fully fit the window
                                                || y[i] + h > SAMPLE_SIZE)
                                            {
                                                overflow = true;
                                                break;
                                            }
                                        }
                                        if(overflow)
                                        {
                                            continue;
                                        }
                                    }


                                    //create the wavelet
                                    std::vector<cv::Rect> rects(K);
                                    for (int i = 0; i < K; i++)
                                    {
                                        rects[i] = cv::Rect(x[i], y[i], w, h);
                                    }

                                    HaarWavelet wavelet(rects, weights);
                                    wavelets.insert( wavelet );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



/**
 * Generates Haar wavelets with 4 rectangles.
 */
void gen4d(WaveletMap &wavelets)
{
    const int K = 4; //number of dimensions of the generated wavelets

    std::vector<float> weights(K);
    weights[0] = 1;
    weights[1] = -1;
    weights[2] = 1;
    weights[3] = -1;

    for(int w = MIN_RECT_WIDTH; w <= SAMPLE_SIZE; w++)
    {
        for(int h = MIN_RECT_HEIGHT; h <= SAMPLE_SIZE; h++)
        {
            int x[K], //x and y positions of each rectangle.
                y[K];

            for(x[0] = 0; x[0] <= SAMPLE_SIZE - w; x[0]+=2) //for each x...
            {
                for(y[0] = 0; y[0] <= SAMPLE_SIZE - h; y[0]+=2) //...and y of the first rectangle...
                {
                    if (   x[0] + w > SAMPLE_SIZE
                        || y[0] + h > SAMPLE_SIZE)
                    {
                        continue;
                    }

                    int dx[K - 1], //dx = horizontal displacement multiplier of the second rectangle.
                        dy[K - 1]; //If bigger than 1 the rectangles will be disjoint. See Pavani's restriction #4.
                                   //dy is similar to dx but in the vertical direction

                    for(dx[0] = -SAMPLE_SIZE / w; dx[0] < SAMPLE_SIZE / w; dx[0]++)
                    {
                        for(dy[0] = -SAMPLE_SIZE / h; dy[0] < SAMPLE_SIZE / h; dy[0]++)
                        {
                            if (dx[0] == 0 && dy[0] == 0)
                            {
                                continue;
                            }

                            x[1] = x[0] + dx[0] * w;
                            y[1] = y[0] + dy[0] * h;

                            if (   x[1] < 0
                                || y[1] < 0
                                || x[1] >= SAMPLE_SIZE
                                || y[1] >= SAMPLE_SIZE
                                || x[1] + w > SAMPLE_SIZE
                                || y[1] + h > SAMPLE_SIZE)
                            {
                                continue;
                            }

                            for(dx[1] = -SAMPLE_SIZE / w; dx[1] < SAMPLE_SIZE / w; dx[1]+=2)
                            {
                                for(dy[1] = -SAMPLE_SIZE / h; dy[1] < SAMPLE_SIZE / h; dy[1]+=2)
                                {
                                    if (dx[1] == 0 && dy[1] == 0)
                                    {
                                        continue;
                                    }

                                    x[2] = x[1] + dx[1] * w;
                                    y[2] = y[1] + dy[1] * h;

                                    if (   x[2] < 0
                                        || y[2] < 0
                                        || x[2] >= SAMPLE_SIZE
                                        || y[2] >= SAMPLE_SIZE
                                        || x[2] + w > SAMPLE_SIZE
                                        || y[2] + h > SAMPLE_SIZE)
                                    {
                                        continue;
                                    }

                                    for(dx[2] = -SAMPLE_SIZE / w; dx[2] < SAMPLE_SIZE / w; dx[2]+=2)
                                    {
                                        for(dy[2] = -SAMPLE_SIZE / h; dy[2] < SAMPLE_SIZE / h; dy[2]+=2)
                                        {
                                            //avoids rectangle overlapping
                                            if ( dx[2] == 0 && dy[2] == 0 )
                                            {
                                                continue;
                                            }

                                            x[3] = x[2] + dx[2] * w;
                                            y[3] = y[2] + dy[2] * h;

                                            if (   x[3] < 0
                                                || y[3] < 0
                                                || x[3] >= SAMPLE_SIZE
                                                || y[3] >= SAMPLE_SIZE
                                                || x[3] + w > SAMPLE_SIZE
                                                || y[3] + h > SAMPLE_SIZE)
                                            {
                                                continue;
                                            }


                                            {//avoids rectangles overlapping
                                                bool overlaps = false;

                                                for (int i = 0; i < K; i++)
                                                {
                                                    for (int j = 0; j < K; j++)
                                                    {
                                                        if (i != j && x[i] == x[j] && y[i] == y[j])
                                                        {
                                                            overlaps = true;
                                                            break;
                                                        }
                                                    }
                                                    if (overlaps)
                                                    {
                                                        break;
                                                    }
                                                }
                                                if (overlaps)
                                                {
                                                    continue;
                                                }
                                            }

                                            {
                                                bool overflow = false;
                                                for (int i = 0; i < K; i++) //...and all rectangles fit into the sampling window...
                                                {
                                                    if(    x[i] < 0
                                                        || y[i] < 0
                                                        || x[i] >= SAMPLE_SIZE //x and y must be at least 1 pixel away from the window's last pixel
                                                        || y[i] >= SAMPLE_SIZE
                                                        || x[i] + w > SAMPLE_SIZE //and the rectangle must fully fit the window
                                                        || y[i] + h > SAMPLE_SIZE)
                                                    {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                                if(overflow)
                                                {
                                                    continue;
                                                }
                                            }


                                            //create the wavelet
                                            std::vector<cv::Rect> rects(K);
                                            for (int i = 0; i < K; i++)
                                            {
                                                rects[i] = cv::Rect(x[i], y[i], w, h);
                                            }

                                            HaarWavelet wavelet(rects, weights);
                                            wavelets.insert( wavelet );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    WaveletMap wavelets;
    {
        gen2d(wavelets);
        const int d2wavelets = wavelets.size();
        std::cout << "Total 2D wavelets generated: " << d2wavelets << std::endl;
        gen3d(wavelets);
        const int d3wavelets = wavelets.size() - d2wavelets;
        std::cout << "Total 3D wavelets generated: " << d3wavelets << std::endl;
        gen4d(wavelets);
        const int d4wavelets = wavelets.size() - d3wavelets - d2wavelets;
        std::cout << "Total 4D wavelets generated: " << d4wavelets << std::endl;
        std::cout << "Wavelets generated: " << wavelets.size() << std::endl;
    }

    //sorts the wavelets
    std::vector<HaarWavelet> sorted;
    {
        WaveletMap::iterator it = wavelets.begin();
        const WaveletMap::iterator end = wavelets.end();
        for(;it != end; ++it)
        {
            sorted.push_back(*it);
        }
        std::sort(sorted.begin(), sorted.end(), wavelet_comparator());
    }

    {
        std::cout << "Writing wavelets to file...";
        writeHaarWavelets(args[1], sorted);
        std::cout << " done." << std::endl;
    }

    return 0;
}
