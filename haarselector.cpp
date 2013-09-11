#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "../libpca-1.2.11/include/pca.h"

#include "../ecrsgen/lib/haarwavelet.h"
#include "../ecrsgen/lib/haarwaveletutilities.h"



bool loadpca(stats::pca &pca, std::string filename)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if (!ifs.is_open())
    {
        return false;
    }

    bool varNumberSet = false;

    while(!ifs.eof())
    {
        std::string line;
        getline(ifs, line);

        std::vector<double> record;
        std::istringstream lineInputStream(line);
        while(!lineInputStream.eof())
        {
            float f;
            if ( lineInputStream >> f )
            {
                record.push_back((double)f);
            }
            else
            {
                break;
            }
        }

        if ( !varNumberSet )
        {
            pca.set_num_variables(record.size());
            varNumberSet = true;
        }

        if (!record.empty())
        {
            pca.add_record(record);
        }
    }

    ifs.close();
}


void printSolution(stats::pca &pca)
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



struct Solution
{
    HaarWavelet * h;
    float variance;

    Solution() : h(0){}
    Solution(const Solution & s) : h(s.h),
                                   variance(s.variance) {}

    Solution &operator=(const Solution & s)
    {
        h = s.h;
        variance = s.variance;
    }
};

bool reverseSolutionSorter(const Solution &s1, const Solution &s2)
{
    return s1.variance < s2.variance;
}


/**
 * Returns the principal component with the smallest variance.
 */
void getPrincipalComponent(stats::pca &pca, Solution & solution)
{
    pca.solve();

    const std::vector<double>::iterator min_value_index =
        std::min_element(pca.get_eigenvalues().begin(), pca.get_eigenvalues().end());
    const std::vector<double> eigenvector = pca.get_eigenvector(min_value_index - pca.get_eigenvalues().begin());

    for(int i = 0; i < eigenvector.size(); ++i)
    {
        solution.h->weight(i, (float) eigenvector[i]);
    }
}



/**
 * From a SRFS dataset, optimize and obtain the best wavelets.
 */
int main(int argc, char * args[])
{
    if (argc != 5)
    {
        return 1;
    }

    boost::filesystem::path inputWaveletsFile = args[1];
    boost::filesystem::path srfsFolder = args[2];
    boost::filesystem::path outputWaveletsFile = args[3];
    int amount_wavelets = 0;
    {
        std::stringstream ss;
        ss << args[4];
        ss >> amount_wavelets;
    }


    cv::Size sampleSize(20, 20);


    std::vector<HaarWavelet *> wavelets;
    loadHaarWavelets(&sampleSize, inputWaveletsFile.native(), wavelets);


    std::vector<Solution> solutions; //the optimal solution for a certain haar wavelet
    solutions.reserve(wavelets.size());


    for (std::vector<HaarWavelet*>::iterator it = wavelets.begin(); it != wavelets.end(); ++it)
    {
        Solution sol; //holds the (optimized) wavelet and the variance of the SRFS's principal component
        sol.h = *it;
        std::stringstream srfsFileName;
        sol.h->write(srfsFileName);
        srfsFileName << ".txt";
        const boost::filesystem::path srfsFile = srfsFolder / srfsFileName.str();

        //prepare the PCA solver to work
        stats::pca pca;
        loadpca(pca, srfsFile.native());
        getPrincipalComponent(pca, sol);
        solutions.push_back(sol);
    }

    //sort the solutions using the variance. The smallest variance goes first
    std::sort(solutions.begin(), solutions.end(), reverseSolutionSorter);

    for(int i = 0; i < wavelets.size(); ++i)
    {
        wavelets[i] = solutions[i].h;
    }

    //write all haar wavelets sorted from best to worst
    writeHaarWavelets(outputWaveletsFile.native().c_str(), wavelets);

    return 0;
}
