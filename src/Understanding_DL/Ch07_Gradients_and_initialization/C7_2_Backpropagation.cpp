/*
 * C7_2_Backpropagation.cpp
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
  //# TO DO -- Replace the line below
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

// Main backward pass routine

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> backward_pass(
		std::vector<torch::Tensor> all_weights, std::vector<torch::Tensor> all_biases,
		std::vector<torch::Tensor> all_f, std::vector<torch::Tensor> all_h, torch::Tensor y, int K) {
	// We'll store the derivatives dl_dweights and dl_dbiases in lists as well
	std::vector<torch::Tensor> all_dl_dweights(K+1);
	std::vector<torch::Tensor> all_dl_dbiases(K+1);

	// And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
	std::vector<torch::Tensor> all_dl_df(K+1);
	std::vector<torch::Tensor> all_dl_dh(K+1);

	// Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

	// Compute derivatives of the loss with respect to the network output
	all_dl_df[K] = d_loss_d_output(all_f[K], y);

	// Now work backwards through the network
	for(int layer = K; layer >= 0; layer += -1) {
	    // TODO Calculate the derivatives of the loss with respect to the biases at layer from all_dl_df[layer]. (eq 7.21)
	    // NOTE!  To take a copy of matrix X, use Z=np.array(X)
	    all_dl_dbiases[layer] = all_dl_df[layer];

	    // TODO Calculate the derivatives of the loss with respect to the weights at layer from
	    // all_dl_df[layer] and all_h[layer] (eq 7.22). Don't forget to use matmul
	    all_dl_dweights[layer] = torch::matmul(all_dl_df[layer], all_h[layer].t());

	    // TODO: calculate the derivatives of the loss with respect to the activations from weight
		// and derivatives of next preactivations (second part of last line of eq 7.24)
	    all_dl_dh[layer] = torch::matmul(all_weights[layer].t(), all_dl_df[layer]);


	    if( layer > 0) {
	    	// TODO Calculate the derivatives of the loss with respect to the pre-activation
	    	// f (use derivative of ReLu function, first part of last line of eq. 7.24)
	    	all_dl_df[layer-1] = indicator_function(all_f[layer - 1]) * all_dl_dh[layer];
	    }
	  }

	  return std::make_tuple(all_dl_dweights, all_dl_dbiases);
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(345);

	// Number of layers
	int K = 5;
	// Number of neurons per layer
	int D = 6;
	// Input layer
	int D_i = 1;
	// Output layer
	int D_o = 1;

	// Make empty lists
	std::vector<torch::Tensor> all_weights(K+1);
	std::vector<torch::Tensor> all_biases(K+1);

	// Create input and output layers
	all_weights[0] = torch::normal(0., 1.0, {D, D_i});
	all_weights[K] = torch::normal(0., 1.0, {D_o, D});
	all_biases[0] = torch::normal(0., 1.0, {D, 1});
	all_biases[K]= torch::normal(0., 1.0, {D_o, 1});

	// Create intermediate layers
	for(auto& layer : range(K-1, 1)) {
	  all_weights[layer] = torch::normal(0., 1.0, {D, D});
	  all_biases[layer] = torch::normal(0., 1.0, {D, 1});
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute network output\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Define input
	torch::Tensor net_input = torch::ones({D_i,1}) * 1.2;
	// Compute network output
	std::vector<torch::Tensor> all_f, all_h;
	torch::Tensor net_output;
	std::tie(net_output, all_f, all_h) = compute_network_output(net_input,all_weights, all_biases);
	printf("True output = %3.3f, Your answer = %3.3f\n", 0.701, net_output[0][0].data().item<float>());

	torch::Tensor y = torch::ones({D_o, 1}) * 20.0;
	torch::Tensor loss = least_squares_loss(net_output, y);
	printf("y = %3.3f Loss = %3.3f\n", y.data().item<float>(), loss.data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Main backward pass routine\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::vector<torch::Tensor> all_dl_dweights, all_dl_dbiases;
	std::tie(all_dl_dweights, all_dl_dbiases) = backward_pass(all_weights, all_biases, all_f, all_h, y, K);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's test if we have the derivatives right using finite differences\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::vector<torch::Tensor> all_dl_dweights_fd(K+1);
	std::vector<torch::Tensor> all_dl_dbiases_fd(K+1);

	float delta_fd = 0.000001;

	// Test the dervatives of the bias vectors
	for(auto& layer : range(K, 0)) {
		torch::Tensor dl_dbias  = torch::zeros_like(all_dl_dbiases[layer]);
	    // For every element in the bias
	    for(auto& row : range(static_cast<int>(all_biases[layer].size(0)), 0)) {
		    // Take copy of biases  We'll change one element each time
	    	std::vector<torch::Tensor> all_biases_copy;
	    	for(auto& a : all_biases) {
	    		all_biases_copy.push_back(a);
	    	}

		    all_biases_copy[layer][row] += delta_fd;
		    std::vector<torch::Tensor> v1, v2;
		    torch::Tensor network_output_1, network_output_2;
		    std::tie(network_output_1, v1, v2) = compute_network_output(net_input, all_weights, all_biases_copy);
		    std::tie(network_output_2, v1, v2) = compute_network_output(net_input, all_weights, all_biases);
		    dl_dbias[row] = (least_squares_loss(network_output_1, y) -
		    					least_squares_loss(network_output_2, y))/delta_fd;
	    }

	    all_dl_dbiases_fd[layer] = dl_dbias;
	    printf("-----------------------------------------------\n");
	    printf("Bias %d, derivatives from backprop:\n", layer);
	    std::cout << all_dl_dbiases[layer] << '\n';
	    printf("Bias %d, derivatives from finite differences\n", layer);
	    std::cout << all_dl_dbiases_fd[layer] << '\n';
	}

	// Test the derivatives of the weights matrices
	for(auto& layer : range(K, 0)) {
		torch::Tensor dl_dweight  = torch::zeros_like(all_dl_dweights[layer]);
	    // For every element in the bias
		for(auto& row : range(static_cast<int>(all_weights[layer].size(0)), 0)) {
			for(auto& col : range(static_cast<int>(all_weights[layer].size(1)), 0)) {
				// Take copy of biases  We'll change one element each time
				std::vector<torch::Tensor> all_weights_copy;
				for(auto& a : all_weights) {
					all_weights_copy.push_back(a);
				}

				all_weights_copy[layer][row][col] += delta_fd;

				std::vector<torch::Tensor> v1, v2;
				torch::Tensor network_output_1, network_output_2;
				std::tie(network_output_1, v1, v2) = compute_network_output(net_input, all_weights_copy, all_biases);
				std::tie(network_output_2, v1, v2) = compute_network_output(net_input, all_weights, all_biases);
				dl_dweight[row][col] = (least_squares_loss(network_output_1, y) -
										least_squares_loss(network_output_2,y))/delta_fd;
			}
		}
		all_dl_dweights_fd[layer] = dl_dweight;
		printf("-----------------------------------------------\n");
		printf("Weight %d, derivatives from backprop:\n", layer);
		std::cout << all_dl_dweights[layer] << '\n';
		printf("Weight %d, derivatives from finite differences\n", layer);
		std::cout << all_dl_dweights_fd[layer] << '\n';
	}

	std::cout << "Done!\n";
	return 0;
}




