#include "mypca.h"

void mypca::solve() {
    assert_num_vars_();

    if (num_records_ < 2)
        throw std::logic_error("Number of records smaller than two.");

    data_.resize(num_records_, num_vars_);

    mean_ = stats::utils::compute_column_means(data_);
    stats::utils::remove_column_means(data_, mean_);

    sigma_ = stats::utils::compute_column_rms(data_);
    if (do_normalize_) stats::utils::normalize_by_column(data_, sigma_);

    arma::Col<double> eigval(num_vars_);
    arma::Mat<double> eigvec(num_vars_, num_vars_);

    cov_mat_ = stats::utils::make_covariance_matrix(data_);
    arma::eig_sym(eigval, eigvec, cov_mat_, solver_.c_str());
    arma::uvec indices = arma::sort_index(eigval, 1);

    for (long i=0; i<num_vars_; ++i) {
        eigval_(i) = eigval(indices(i));
        eigvec_.col(i) = eigvec.col(indices(i));
    }

    stats::utils::enforce_positive_sign_by_column(eigvec_);
    proj_eigvec_ = eigvec_;

    princomp_ = data_ * eigvec_;

    energy_(0) = arma::sum(eigval_);
    eigval_ *= 1./energy_(0);

    if (do_bootstrap_) bootstrap_eigenvalues_();
}
