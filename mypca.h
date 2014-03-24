#ifndef MYPCA_H
#define MYPCA_H

#include "pca.h"
#include <armadillo>



/**
 * Extention of libpca's stats::pca class that adds the possibility
 * of easily extracting the covariance matrix after the PCA procedure.
 */
class mypca : public stats::pca //99% copy and paste
{
public:
    arma::Mat<double> cov_mat_;
    void solve();
};



#endif // MYPCA_H
