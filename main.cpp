#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include "/home/ramiro/workspace/libpca-1.2.11/include/pca.h"

using namespace std;


bool loadpca(stats::pca &pca, char* filename)
{
    std::ifstream ifs;
    ifs.open(filename, std::ifstream::in);

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




int main(int argc, char * args[])
{
    if (argc != 2)
    {
        return 1;
    }

    {
        //
    }

    {
        stats::pca pca;
        loadpca(pca, args[1]);
        std::cout << "Loaded SRFS..." << std::endl;


        pca.solve();


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

    return 0;
}
