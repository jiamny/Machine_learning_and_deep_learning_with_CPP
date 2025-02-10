/*
 * PrincipalComponentsAnalysis.hpp
 *
 *  Created on: May 4, 2024
 *      Author: jiamny
 */

#ifndef PRINCIPALCOMPONENTSANALYSIS_HPP_
#define PRINCIPALCOMPONENTSANALYSIS_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>

using torch::indexing::Slice;
using torch::indexing::None;

class PCA {
public:
	PCA(int64_t n_components) {
        //:param n_components: Number of principal components the data should be reduced too.
        components = n_components;
	}

    torch::Tensor fit_transform(torch::Tensor X) {
        /*
        * Centering our inputs with mean
        * Finding covariance matrix using centered tensor
        * Finding eigen value and eigen vector using torch.eig()
        * Sorting eigen values in descending order and finding index of high eigen values
        * Using sorted index, get the eigen vectors
        * Tranforming the Input vectors with n columns into PCA components with reduced dimension
        :param X: Input tensor with n columns.
        :return: Output tensor with reduced principal components
        */
    	int64_t n_samples = X.size(0);
    	X = X.to(torch::kDouble);
    	c10::OptionalArrayRef<long int> dm = {0};
    	torch::Tensor centering_X = X - X.mean(dm);
    	torch::Tensor covariance_matrix = torch::mm(centering_X.t(), centering_X)/(n_samples - 1);

    	torch::Tensor eigvals, eigvectors;
        // 对协方差矩阵进行特征值分解
        std::tie(eigvals, eigvectors) = torch::linalg_eig(covariance_matrix);
        eigvals = eigvals.to(torch::kDouble);
        eigvectors = eigvectors.to(torch::kDouble);
        auto idx = eigvals.argsort(-1, true);

        auto eigenvalues = torch::index_select(eigvals, 0, idx ).index({Slice(0, components)});
        auto eigenvectors = torch::index_select(eigvectors, 1, idx ).index({Slice(), Slice(0, components)});
        auto transformed = torch::mm(eigenvectors.t(), centering_X.t()).t();
        return transformed;
    }
private:
    int64_t components;
};

#endif /* PRINCIPALCOMPONENTSANALYSIS_HPP_ */
