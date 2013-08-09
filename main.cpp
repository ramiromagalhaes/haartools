#include <iostream>

#include "/home/ramiro/workspace/libpca-1.2.11/include/pca.h"

using namespace std;



int main(int argc, char * args[])
{
    stats::pca pca;

    {
        std::ifstream ifs;
        ifs.open(args[1], std::ifstream::in);

        while(!ifs.eof())
        {
            vector<double> record;
            while(ifs.get() != '\n' || ifs.get() != 0)
            {
                float data;
                ifs >> data;
                record.push_back((double)data);
            }
            pca.add_record(record);
        }

        ifs.close();
    }

    std::cout << "Loaded PCA..." << std::endl;



    pca.solve();



    std::cout << "Energy = "
              << pca.get_energy()
              << " ("
              << stats::utils::get_sigma(pca.get_energy_boot())
              << ")"
              << std::endl;

    const std::vector<double> eigenvalues = pca.get_eigenvalues();

    std::cout << "First three eigenvalues = "
              << eigenvalues[0] << ", "
              << eigenvalues[1] << ", "
              << eigenvalues[2] << std::endl;

    cout << "Orthogonal Check = " << pca.check_eigenvectors_orthogonal() << std::endl;
    cout << "Projection Check = " << pca.check_projection_accurate() << std::endl;

    pca.save("pca_results");

    return 0;
}
