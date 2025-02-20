/*
 * UDL_util.h
 *
 *  Created on: Jan 9, 2025
 *      Author: jiamny
 */

#ifndef SRC_UTILS_UDL_UTIL_H_
#define SRC_UTILS_UDL_UTIL_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

using torch::indexing::Slice;
using torch::indexing::None;

// ------------------------------------------------------------------------
// Chapter 6-3 Stochastic_Gradient_Descent & Chapter 9-1 L2_Regularization
// ------------------------------------------------------------------------

// Gabor model definition
torch::Tensor model(torch::Tensor phi, torch::Tensor x);

void draw_model(torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor),
		torch::Tensor phi, std::string title="");

torch::Tensor compute_loss(torch::Tensor data_x, torch::Tensor data_y,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi);

// These came from writing out the expression for the sum of squares loss and taking the
// derivative with respect to phi0 and phi1. It was a lot of hassle to get it right!
torch::Tensor gabor_deriv_phi0(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi0, torch::Tensor phi1);

torch::Tensor gabor_deriv_phi1(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi0, torch::Tensor phi1);

torch::Tensor compute_gradient(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi);

void draw_loss_function(torch::Tensor (*compute_loss)(torch::Tensor, torch::Tensor,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor),
		torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi_iters = torch::empty(0));

torch::Tensor gradient_descent_step(torch::Tensor phi, torch::Tensor data, float learning_rate,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor));

// ------------------------------------------------------------------------
// Chapter 9-3 Ensembling and Chapter 9-4 Bayesian approach
// ------------------------------------------------------------------------
// The true function that we are trying to estimate, defined on [0,1]
torch::Tensor true_function(torch::Tensor x);

std::tuple<torch::Tensor, torch::Tensor> generate_data(int n_data, float sigma_y=0.3);

void plot_function(torch::Tensor x_func, torch::Tensor y_func, torch::Tensor x_data=torch::empty(0),
		torch::Tensor y_data=torch::empty(0), torch::Tensor sigma_func=torch::empty(0), std::string tlt="",
		torch::Tensor x_model=torch::empty(0), torch::Tensor y_model=torch::empty(0),
		torch::Tensor sigma_model=torch::empty(0));

torch::Tensor _network(torch::Tensor x, torch::Tensor beta, torch::Tensor omega);

// ------------------------------------------------------------------------
// Chapter 11.1 linear cross correlation
// ------------------------------------------------------------------------
std::vector<double> linear_cross_correlation(std::vector<double>  a, std::vector<double> v, std::string="same");

#endif /* SRC_UTILS_UDL_UTIL_H_ */
