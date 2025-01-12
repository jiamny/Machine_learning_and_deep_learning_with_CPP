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


#endif /* SRC_UTILS_UDL_UTIL_H_ */
