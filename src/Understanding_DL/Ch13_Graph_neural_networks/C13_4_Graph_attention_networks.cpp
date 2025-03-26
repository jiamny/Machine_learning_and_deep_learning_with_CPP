/*
 * C13_4_Graph_attention_networks.cpp
 *
 *  Created on: Mar 20, 2025
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

// Define softmax operation that works independently on each column
torch::Tensor softmax_cols(torch::Tensor data_in) {
    // Exponentiate all of the values
	torch::Tensor exp_values = torch::exp(data_in);

    // Sum over columns
	c10::OptionalArrayRef<long int> dim = {0};
	torch::Tensor denom = torch::sum(exp_values, dim);

    // Compute softmax (numpy broadcasts denominator to all rows automatically)
	torch::Tensor softmax = exp_values / denom;
    // return the answer
  return softmax;
}

// Define the Rectified Linear Unit (ReLU) function
torch::Tensor _ReLU(torch::Tensor preactivation) {
	const std::optional<c10::Scalar> min = {0.0};
	torch::Tensor activation = preactivation.clip(min);
    return activation;
}

// Now let's compute self attention in matrix form
torch::Tensor graph_attention(torch::Tensor X, torch::Tensor omega,
							torch::Tensor beta, torch::Tensor phi, torch::Tensor A) {
 /*
 # Write this function (see figure 13.12c)
 # 1. Compute X_prime
 # 2. Compute S
 # 3. To apply the mask, set S to a very large negative number (e.g. -1e20) everywhere where A+I is zero
 # 4. Run the softmax function to compute the attention values
 # 5. Postmultiply X' by the attention values
 # 6. Apply the ReLU function
 */
	torch::Tensor output;
	int N = A.size(0);
	torch::Tensor t1 = torch::ones({N, 1});
	torch::Tensor I = torch::eye(N).to(torch::kInt32);
	torch::Tensor X_prime = torch::matmul(beta, t1.t()) + torch::matmul(omega, X);

	torch::Tensor S = torch::zeros({N, N});

	for(auto& m : range(N, 0)) {
	    for(auto& n : range(N, 0)) {
	    	torch::Tensor X_p = torch::concat({X_prime.index({Slice(), m}), X_prime.index({Slice(), n})}, 0).reshape({-1, 1});
	    	torch::Tensor Smn = _ReLU(torch::sum(phi.t() * X_p));
	        S[m][n] =  Smn.data().item<float>();
	    }
	}
	torch::Tensor AI = A + I;
	S.masked_fill_(AI == 0, -1e20);
	output = _ReLU(torch::matmul(X_prime, softmax_cols(S)));

	return output;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	// Number of nodes in the graph
	int N = 8;
	// Number of dimensions of each input
	int D = 4;

	// Define a graph
	torch::Tensor A = torch::tensor(
				 {{0,1,0,1,0,0,0,0},
	              {1,0,1,1,1,0,0,0},
	              {0,1,0,0,1,0,0,0},
	              {1,1,0,0,1,0,0,0},
	              {0,1,1,1,0,1,0,1},
	              {0,0,0,0,1,0,1,1},
	              {0,0,0,0,0,1,0,0},
	              {0,0,0,0,1,1,0,0}});
	print(A);
	std::cout << '\n';

	// Let's also define some random data
	torch::Tensor X = torch::randn({D,N});


	// Choose random values for the parameters
	torch::Tensor omega = torch::randn({D,D});
	torch::Tensor beta = torch::randn({D,1});
	torch::Tensor phi = torch::randn({1,2*D});

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Test out the graph attention mechanism\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor output = graph_attention(X, omega, beta, phi, A);
	printf("Answer is:\n");
	print(output);

	std::cout << "\nDone!\n";
	return 0;
}




