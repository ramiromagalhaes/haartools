#include <iostream>
#include <string>
#include <sstream>
#include <limits>

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



std::vector<float> getPrincipalComponent(stats::pca &pca)
{
    pca.solve();

    int max_value_index = 0;
    float curr_max = std::numeric_limits<float>::min();
    for (int i = 0; i < pca.get_num_variables(); ++i)
    {
        if (pca.get_eigenvalue(i) > curr_max)
        {
            curr_max = pca.get_eigenvalue(i);
            max_value_index = i;
        }
    }

    const std::vector<double> eigenvector = pca.get_eigenvector(max_value_index);
    std::vector<float> returnMe;
    for(std::vector<double>::const_iterator it = eigenvector.begin(); it != eigenvector.end(); ++it)
    {
        const double d = *it;
        returnMe.push_back( (float)d );
    }
    return returnMe;
}



int main(int argc, char * args[])
{
    if (argc != 4)
    {
        return 1;
    }

    boost::filesystem::path inputWaveletsFile = args[1];
    boost::filesystem::path srfsFolder = args[2];
    boost::filesystem::path outputWaveletsFile = args[3];


    cv::Size sampleSize(20, 20);
    cv::Point position(0, 0);

    std::vector<HaarWavelet *> wavelets;
    loadHaarWavelets(&sampleSize, &position, inputWaveletsFile.native(), wavelets);

    //carrega os haar wavelets de um arquivo
    //para cada haar wavelet carrega um srfs de acordo com a lista de haar wavelets
    //atualiza os pesos do haar wavelet
    //escreve no arquivo de saída

    for (std::vector<HaarWavelet*>::iterator it = wavelets.begin(); it != wavelets.end(); ++it)
    {
        HaarWavelet * wavelet = *it;
        std::stringstream srfsFileName;
        wavelet->write(srfsFileName);
        srfsFileName << ".txt";
        const boost::filesystem::path srfsFile = srfsFolder / srfsFileName.str();

        std::cout << "Optimizing Haar Wavelet with file " << srfsFile.string() << "...";

        stats::pca pca;
        loadpca(pca, srfsFile.native());
        const std::vector<float> newWeights = getPrincipalComponent(pca);
        for (int i = 0; i < newWeights.size(); ++i)
        {
            wavelet->weight(i, newWeights[i]);
        }

        std::cout << "done." << std::endl;
    }

    writeHaarWavelets(outputWaveletsFile.native().c_str(), wavelets);

    return 0;
}
