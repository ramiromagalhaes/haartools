#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

#include "haarwavelet.h"



class ProbabilisticClassifierData : public HaarWavelet
{
public:
    ProbabilisticClassifierData() : histogram(100) {}

    ProbabilisticClassifierData& operator=(const ProbabilisticClassifierData & c)
    {
        rects = c.rects;
        weights = c.weights;
        mean = c.mean;
        stdDev = c.stdDev;
        histogram = c.histogram;

        return *this;
    }

    bool operator < (const ProbabilisticClassifierData & rh) const
    {
        return stdDev < rh.stdDev;
    }

    bool read(std::istream &input)
    {
        HaarWavelet::read(input);

        input >> mean
              >> stdDev;

        int i = 0;
        for (; i < 100; ++i)
        {
            input >> histogram[i];
        }

        if (i == 100)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    std::vector<double>& getHistogram()
    {
        return histogram;
    }

private:
    double stdDev; //statistics taken from the feature value, not directly from the SRFS
    double mean;
    std::vector<double> histogram;
};



/**
 * Loads many WeakHypothesis found in a file to a vector of HaarClassifierType.
 */
bool loadClassifierData(const std::string &filename, std::vector<ProbabilisticClassifierData> &classifiers)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        std::string line;
        getline(ifs, line);
        if (line.empty())
        {
            break;
        }
        std::istringstream lineInputStream(line);

        ProbabilisticClassifierData classifier;
        classifier.read(lineInputStream);

        if ( !ifs.eof() )
        {
            classifiers.push_back( classifier );
        }
        else
        {
            break;
        }
    } while (true);

    ifs.close();

    return true;
}



/**
 * Checks if the Haar-like classifiers produced by
 * haaroptimizer2.cpp makes sense.
 */
int main(int argc, char * args[])
{
    if (argc != 2) {
        return 1;
    }

    std::vector<ProbabilisticClassifierData> classifiers;
    loadClassifierData(args[1], classifiers);

    std::vector<ProbabilisticClassifierData>::iterator it = classifiers.begin();
    const std::vector<ProbabilisticClassifierData>::iterator end = classifiers.end();

    bool ok = true;

    for(; it != end; ++it)
    {
        std::vector<double> histogram = it->getHistogram();

        double sumOfHistogram = std::accumulate(histogram.begin(), histogram.end(), .0);
        if ( std::fabs(sumOfHistogram - 1.0) > 0.000001)
        {
            ok = false;
            std::cerr << "Haar-like classifier histogram with index " << it - classifiers.begin() << " adds to " << sumOfHistogram << std::endl;
        }
    }

    if (ok)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
