/*
 * C9_4_Bayesian_approach.cpp
 *
 *  Created on: Jan 10, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/UDL_util.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;


torch::Tensor compute_H(torch::Tensor x_data, int n_hidden) {
	torch::Tensor psi1 = torch::ones({n_hidden+1, 1});
	torch::Tensor psi0 = torch::linspace(0.0, 1.0, n_hidden) * -1;

	  int n_data = x_data.size(0);
	  // First compute the hidden variables
	  torch::Tensor H = torch::ones({n_hidden+1, n_data});
	  for(auto& i : range(n_hidden, 0)) {
	    for(auto& j : range(n_data, 0)) {
	      // Compute preactivation
	      H.index_put_({i, j}, psi1[i] * x_data[j] + psi0[i]);
	      // Apply ReLU to get activation
	      if( H[i][j].data().item<float>() < 0)
	    	  H[i][j] = 0;
	    }
	  }

	  return H;
}

std::tuple<torch::Tensor, torch::Tensor> compute_param_mean_covar(torch::Tensor x_data,
										torch::Tensor y_data, int n_hidden, float sigma_sq, float sigma_p_sq) {

	// Retrieve the matrix containing the hidden variables
	torch::Tensor H = compute_H(x_data, n_hidden);
	torch::Tensor Hy = torch::matmul(H, y_data);

	torch::Tensor HHT = torch::matmul(H, H.t());
	torch::Tensor I = torch::eye(n_hidden+1);

	// -- Compute the covariance matrix (you will need np.transpose(), np.matmul(), np.linalg.inv())
	torch::Tensor phi_covar = torch::inverse(((1.0/sigma_sq)*HHT + (1.0/sigma_p_sq)*I));

	// -- Compute the mean matrix
	torch::Tensor phi_mean = torch::matmul((1.0/sigma_sq)*phi_covar, Hy);

	return std::make_tuple(phi_mean, phi_covar);
}

// Predict mean and variance of y_star from x_star
std::tuple<torch::Tensor, torch::Tensor> inference (torch::Tensor x_star, torch::Tensor x_data, torch::Tensor y_data,
		float sigma_sq, float sigma_p_sq, int n_hidden) {

    // Compute hidden variables
	torch::Tensor h_star = compute_H(x_star, n_hidden);
	torch::Tensor H = compute_H(x_data, n_hidden);
	torch::Tensor Hy = torch::matmul(H, y_data);
	torch::Tensor HHT = torch::matmul(H, H.t());
	torch::Tensor I = torch::eye(n_hidden+1);
	torch::Tensor hT1 = h_star.t(); //torch::cat({h_star.t(), torch::tensor({{1}})}, 1);
	torch::Tensor h1 = h_star; 		//torch::cat({h_star, torch::tensor({{1}})}, 0);
	torch::Tensor phi_covar = torch::inverse(((1.0/sigma_sq)*HHT + (1.0/sigma_p_sq)*I));

    // Compute mean and variance of y*
	torch::Tensor y_star_var =  torch::matmul(torch::matmul(hT1, phi_covar), h1);
	torch::Tensor y_star_mean = torch::matmul(torch::matmul((1.0/sigma_sq)*hT1, phi_covar), Hy);

    return std::make_tuple(y_star_mean, y_star_var);
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw the fitted function, together with uncertainty used to generate points\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Generate true function
	torch::Tensor x_func = torch::linspace(0, 1.0, 100);
	torch::Tensor y_func = true_function(x_func);

	// Generate some data points

	float sigma_func = 0.3;
	int n_data = 15;
	torch::Tensor x_data, y_data;
	std::tie(x_data, y_data)= generate_data(n_data, sigma_func);
	std::cout << x_data.sizes() << " " << y_data.sizes() << '\n';


	// Plot the function, data and uncertainty
	plot_function(x_func, y_func, x_data, y_data, torch::tensor({sigma_func}), "data and uncertainty");

	// Define parameters
	int n_hidden = 5;
	float sigma_sq = sigma_func * sigma_func;
	// Arbitrary large value reflecting the fact we are uncertain about the
	// parameters before we see any data
	int sigma_p_sq = 1000;

	// Compute the mean and covariance matrix
	torch::Tensor phi_mean, phi_covar;
	std::tie(phi_mean, phi_covar) = compute_param_mean_covar(x_data, y_data, n_hidden, sigma_sq, sigma_p_sq);
	std::cout << phi_covar.sizes() << '\n';
	std::cout << phi_mean.sizes() << '\n';

	// Let's draw the mean model
	torch::Tensor x_model = x_func;
	torch::Tensor y_model_mean = _network(x_model, phi_mean.index({-1}), phi_mean.index({Slice(0, n_hidden)}));
	plot_function(x_func, y_func, x_data, y_data, torch::empty(0), "the mean model", x_model, y_model_mean);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw two samples from the normal distribution over the parameters\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	MultivariateNormalx mvn = MultivariateNormalx(phi_mean, phi_covar);
	torch::Tensor phi_sample1 = mvn.rsample();
	torch::Tensor phi_sample2 = mvn.rsample();
	std::cout << phi_sample1.sizes() << '\n';

	// Run the network for these two sample sets of parameters
	torch::Tensor y_model_sample1 = _network(x_model, phi_sample1.index({-1}), phi_sample1.index({Slice(0, n_hidden)}));
	torch::Tensor y_model_sample2 = _network(x_model, phi_sample2.index({-1}), phi_sample2.index({Slice(0, n_hidden)}));

	// Draw the two models
	plot_function(x_func, y_func, x_data, y_data, torch::empty(0), "phi sample1", x_model, y_model_sample1);
	plot_function(x_func, y_func, x_data, y_data, torch::empty(0), "phi sample2", x_model, y_model_sample2);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Inference example\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	x_model = x_func;
	torch::Tensor y_model = torch::zeros_like(x_model);
	torch::Tensor y_model_std = torch::zeros_like(x_model);
	for(auto& c_model : range(static_cast<int>(x_model.size(0)), 0)) {
		torch::Tensor y_star_mean, y_star_var;
		std::tie(y_star_mean, y_star_var) = inference(x_model[c_model]*torch::ones({1,1}),
														x_data, y_data, sigma_sq, sigma_p_sq, n_hidden);

	  y_model.index_put_({c_model}, y_star_mean);
	  y_model_std.index_put_({c_model}, torch::sqrt(y_star_var));
	}

	// Draw the model
	plot_function(x_func, y_func, x_data, y_data, torch::empty(0), "inference", x_model, y_model, y_model_std);

	std::cout << "Done!\n";
	return 0;
}




