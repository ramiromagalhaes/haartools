#ifndef MYPCA_H
#define MYPCA_H

#include "pca.h"
#include <armadillo>



class mypca : public stats::pca //99% copy and paste
{
public:
    arma::Mat<double> cov_mat_;
    void solve();
};



#endif // MYPCA_H
