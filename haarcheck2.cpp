#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

#include "haarwavelet.h"


#define HISTOGRAM_BUCKETS 12


class ProbabilisticClassifierData : public DualWeightHaarWavelet
{
public:
    ProbabilisticClassifierData() : histogram(HISTOGRAM_BUCKETS) {}

    ProbabilisticClassifierData& operator=(const ProbabilisticClassifierData & c)
    {
        rects = c.rects;
        weightsPositive = c.weightsPositive;
        weightsNegative = c.weightsNegative;
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
        if ( !DualWeightHaarWavelet::read(input) )
        {
            return false;
        }

        input >> mean
              >> stdDev;

        int buckets = 0;
        input >> buckets;
        for (int i = 0; i < buckets; ++i)
        {
            input >> histogram[i];
        }

        return true;
    }

    std::vector<double>& getHistogram()
    {
        return histogram;
    }

    double getStdDev() const
    {
        return stdDev;
    }

    double getMean() const
    {
        return mean;
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
        if ( std::fabs(sumOfHistogram - 1.0) > 0.000001 )
        {
            ok = false;
            std::cerr << "Haar-like feature with index " << it - classifiers.begin() << " histogram adds to " << sumOfHistogram << std::endl;
        }

        if ( std::fabs(it->getStdDev()) < 0.000009 )
        {
            ok = false;
            std::cerr << "Haar-like feature with index " << it - classifiers.begin() << " standard deviation is 0.";
        }
    }

    std::cout << "Total Haar-like features tested: " << end - classifiers.begin() << std::endl;

    if (ok)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
