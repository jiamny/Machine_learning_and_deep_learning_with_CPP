/*
 * C7_3_Initialization.cpp
 *
 *  Created on: Dec 16, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> init_params(int K, int D, float sigma_sq_omega) {
	// Set seed so we always get the same random numbers
	torch::manual_seed(0);

	// Input layer
	int D_i = 1;
	// Output layer
	int D_o = 1;

	// Make empty lists
	std::vector<torch::Tensor> all_weights(K+1);
	std::vector<torch::Tensor> all_biases(K+1);

	// Create input and output layers
	all_weights[0] = torch::normal(0., 1.0, {D, D_i})*std::sqrt(sigma_sq_omega);
	all_weights[K] = torch::normal(0., 1.0, {D_o, D}) * std::sqrt(sigma_sq_omega);
	all_biases[0] = torch::zeros({D, 1});
	all_biases[K]= torch::zeros({D_o, 1});

	// Create intermediate layers
	for(auto& layer : range(K - 1, 1)) {
	    all_weights[layer] = torch::normal(0., 1.0, {D, D})*std::sqrt(sigma_sq_omega);
	    all_biases[layer] = torch::zeros({D, 1});
	}

	return std::make_tuple(all_weights, all_biases);
}

// Define the Rectified Linear Unit (ReLU) function
torch::Tensor ReLU(torch::Tensor x) {
	if( x.dim() == 1)
		x = x.reshape({1, -1});

	return torch::where(x >=0.0, x, 0.0);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>> compute_network_output(
		torch::Tensor net_input, std::vector<torch::Tensor> all_weights, std::vector<torch::Tensor>all_biases) {

  // Retrieve number of layers
  int K = all_weights.size() - 1;

  // We'll store the pre-activations at each layer in a list "all_f"
  // and the activations in a second list "all_h".
  std::vector<torch::Tensor> all_f(K+1);
  std::vector<torch::Tensor> all_h(K+1);

  // For convenience, we'll set
  // all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input;

  // Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
  for(auto& layer : range(K, 0)) {
      // Update preactivations and activations at this layer according to eqn 7.16
      // Remember to use np.matmul for matrix multiplications
      all_f[layer] = all_biases[layer] + torch::matmul(all_weights[layer], all_h[layer]);
      all_h[layer+1] = ReLU(all_f[layer]);
  }

  // Compute the output from the last hidden layer
  //# -- Replace the line below
  all_f[K] = all_biases[K] + torch::matmul(all_weights[K], all_h[K]);

  // Retrieve the output
  torch::Tensor net_output = all_f[K].clone();

  return std::make_tuple(net_output, all_f, all_h);
}

torch::Tensor least_squares_loss(torch::Tensor net_output, torch::Tensor y) {
  return torch::sum(torch::pow(net_output - y, 2));
}

torch::Tensor d_loss_d_output(torch::Tensor net_output, torch::Tensor y) {
    return 2*(net_output -y);
}

// We'll need the indicator function
torch::Tensor indicator_function(torch::Tensor x_in) {
  x_in.masked_fill_(x_in >= 0, 1);
  x_in.masked_fill_(x_in < 0, 0);
  return x_in;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
backward_pass(std::vector<torch::Tensor> all_weights, std::vector<torch::Tensor> all_biases,
				std::vector<torch::Tensor> all_f, std::vector<torch::Tensor> all_h, torch::Tensor y) {
	// Retrieve number of layers
	int K = all_weights.size() - 1;

	// We'll store the derivatives dl_dweights and dl_dbiases in lists as well
	std::vector<torch::Tensor> all_dl_dweights(K+1);
	std::vector<torch::Tensor> all_dl_dbiases(K+1);

	// And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
	std::vector<torch::Tensor> all_dl_df(K+1);
	std::vector<torch::Tensor> all_dl_dh(K+1);

	// Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output
	// Compute derivatives of net output with respect to loss
	  all_dl_df[K] = d_loss_d_output(all_f[K], y);

	// Now work backwards through the network
	for(int layer = K; layer >= 0; layer += -1) {
	    // Calculate the derivatives of biases at layer from all_dl_df[K]. (eq 7.13, line 1)
	    all_dl_dbiases[layer] = all_dl_df[layer];

	    // Calculate the derivatives of weight at layer from all_dl_df[K] and all_h[K] (eq 7.13, line 2)
	    all_dl_dweights[layer] = torch::matmul(all_dl_df[layer], all_h[layer].t());

	    // Calculate the derivatives of activations from weight and derivatives of next preactivations (eq 7.13, line 3 second part)
	    all_dl_dh[layer] = torch::matmul(all_weights[layer].t(), all_dl_df[layer]);

	    // Calculate the derivatives of the pre-activation f with respect to activation h (eq 7.13, line 3, first part)
	    if( layer > 0 )
	      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer];
	}

	return std::make_tuple(all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// 5 layers with 8 neurons per layer\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// # Number of layers
	int K = 5;
	// Number of neurons per layer
	int D = 8;
	// Input layer
	int D_i = 1;
	// Output layer
	int D_o = 1;
	// Set variance of initial weights to 1
	float sigma_sq_omega = 1.0;
	// Initialize parameters
	std::vector<torch::Tensor> all_weights, all_biases;
	std::tie(all_weights, all_biases) = init_params(K, D, sigma_sq_omega);

	int n_data = 1000;
	torch::Tensor data_in = torch::normal(0.0, 1.0, {1, n_data});
	std::vector<torch::Tensor> all_f, all_h;
	torch::Tensor net_output;
	std::tie(net_output, all_f, all_h) = compute_network_output(data_in, all_weights, all_biases);

	for(auto& layer : range(K, 1)) {
	  printf("Layer %d, std of hidden units = %3.3f\n", layer, torch::std(all_h[layer]).data().item<float>());
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Change this to 10 layers with 40 hidden units per layer\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	K = 10;
	D = 40;
	std::tie(all_weights, all_biases) = init_params(K, D, sigma_sq_omega);

	data_in = torch::normal(0.0, 1.0, {1, n_data});
	std::tie(net_output, all_f, all_h) = compute_network_output(data_in, all_weights, all_biases);

	for(auto& layer : range(K, 1)) {
		printf("Layer %d, std of hidden units = %3.3lf\n", layer, torch::std(all_h[layer]).data().item<double>());
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// 5 layers with 8 neurons per layer, the gradients of the hidden units\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Number of layers
	K = 5;
	// Number of neurons per layer
	D = 8;
	// Input layer
	D_i = 1;
	// Output layer
	D_o = 1;
	// Set variance of initial weights to 1
	sigma_sq_omega = 1.0;
	// Initialize parameters
	std::tie(all_weights, all_biases) = init_params(K, D, sigma_sq_omega);

	// For simplicity we'll just consider the gradients of the weights and biases between the first and last hidden layer
	n_data = 100;
	std::vector<torch::Tensor> aggregate_dl_df(K+1);
	for(auto& layer : range(K - 1, 1)) {
		// These 3D arrays will store the gradients for every data point
	    aggregate_dl_df[layer] = torch::zeros({D, n_data});
	}

	// We'll have to compute the derivatives of the parameters for each data point separately
	for(auto& c_data : range(n_data, 0)) {
	    data_in = torch::normal(0., 1., {1, 1});
	    torch::Tensor y = torch::zeros({1, 1});
	    std::tie(net_output, all_f, all_h) = compute_network_output(data_in, all_weights, all_biases);

	    std::vector<torch::Tensor> all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df;
	    std::tie(all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df) = backward_pass(all_weights, all_biases, all_f, all_h, y);
	    for(auto& layer : range(K - 1, 1)) {
	        aggregate_dl_df[layer].index_put_({Slice(),c_data}, torch::squeeze(all_dl_df[layer]));
	    }
	}

	for(auto& layer : range(K-1, 1)) {
	  printf("Layer %d, std of dl_dh = %3.3lf\n", layer, torch::std(aggregate_dl_df[layer].ravel()).data().item<double>());
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// 10 layers with 40 neurons per layer, the gradients of the hidden units\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	K = 10;
	D = 40;
	std::tie(all_weights, all_biases) = init_params(K, D, sigma_sq_omega);
	std::tie(all_weights, all_biases) = init_params(K, D, sigma_sq_omega);

	// For simplicity we'll just consider the gradients of the weights and biases between the first and last hidden layer
	n_data = 100;
	std::vector<torch::Tensor> aggregate_dl_df_2(K+1);
	for(auto& layer : range(K - 1, 1)) {
		// These 3D arrays will store the gradients for every data point
	    aggregate_dl_df_2[layer] = torch::zeros({D, n_data});
	}

	// We'll have to compute the derivatives of the parameters for each data point separately
	for(auto& c_data : range(n_data, 0)) {
	    data_in = torch::normal(0., 1., {1, 1});
	    torch::Tensor y = torch::zeros({1, 1});
	    std::tie(net_output, all_f, all_h) = compute_network_output(data_in, all_weights, all_biases);

	    std::vector<torch::Tensor> all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df;
	    std::tie(all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df) = backward_pass(all_weights, all_biases, all_f, all_h, y);
	    for(auto& layer : range(K - 1, 1)) {
	        aggregate_dl_df_2[layer].index_put_({Slice(),c_data}, torch::squeeze(all_dl_df[layer]));
	    }
	}

	for(auto& layer : range(K-1, 1)) {
	  printf("Layer %d, std of dl_dh = %3.3lf\n", layer, torch::std(aggregate_dl_df_2[layer].ravel()).data().item<double>());
	}

	std::cout << "Done!\n";
	return 0;
}




