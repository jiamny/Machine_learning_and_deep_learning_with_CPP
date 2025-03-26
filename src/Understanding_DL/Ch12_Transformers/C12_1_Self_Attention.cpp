/*
 * C12_1_Self_Attention.cpp
 *
 *  Created on: Mar 3, 2025
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


// Now let's compute self attention in matrix form
torch::Tensor self_attention(torch::Tensor X, torch::Tensor omega_v, torch::Tensor omega_q,
		torch::Tensor omega_k, torch::Tensor beta_v, torch::Tensor beta_q, torch::Tensor beta_k) {

	// 1. Compute queries, keys, and values
	torch::Tensor Q = beta_q + torch::matmul(omega_q, X);
	torch::Tensor K = beta_k + torch::matmul(omega_k, X);
	torch::Tensor V = beta_v + torch::matmul(omega_v, X);

	// 2. Compute dot products
	torch::Tensor A = torch::matmul(K.t(), Q);

	// 3. Apply softmax to calculate attentions
	torch::Tensor Aw = softmax_cols(A);

	// 4. Weight values by attentions
	torch::Tensor x_prime = torch::matmul(V, Aw);
	return x_prime;
}

//  Now let's compute self attention in matrix form
torch::Tensor scaled_dot_product_self_attention(torch::Tensor X, torch::Tensor omega_v, torch::Tensor omega_q,
		torch::Tensor omega_k, torch::Tensor beta_v, torch::Tensor beta_q, torch::Tensor beta_k) {
	// 1. Compute queries, keys, and values
	torch::Tensor Q = beta_q + torch::matmul(omega_q, X);
	torch::Tensor K = beta_k + torch::matmul(omega_k, X);
	torch::Tensor V = beta_v + torch::matmul(omega_v, X);

	// 2. Compute dot products
	torch::Tensor A = torch::matmul(K.t(), Q);

    // 3. Scale the dot products as in equation 12.9
	int Dq = Q.size(1);
	A = A/std::ceil(std::sqrt(Dq *1.0));

	// 3. Apply softmax to calculate attentions
	torch::Tensor Aw = softmax_cols(A);

	// 4. Weight values by attentions
	torch::Tensor x_prime = torch::matmul(V, Aw);
	return x_prime;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	torch::Tensor X = torch::tensor({
		{1.78862847, -0.2773882,  -0.04381817},
		{0.43650985, -0.35475898, -0.47721803},
		{0.09649747, -0.08274148, -1.31386475},
		{-1.8634927,  -0.62700068,  0.88462238}});


	torch::Tensor omega_q = torch::tensor(
			{{1.76405235,  0.40015721,  0.9787379,  2.2408932 },
			{1.86755799, -0.97727788,  0.95008842, -0.15135721},
			 {-0.10321885,  0.4105985,   0.14404357,  1.45427351},
			 {0.76103773,  0.12167502,  0.44386323,  0.33367433}});

	torch::Tensor omega_k = torch::tensor(
			{{ 1.49407907, -0.20515826,  0.3130677,  -0.85409574},
			 {-2.55298982,  0.6536186,   0.8644362,  -0.74216502},
			 { 2.26975462, -1.45436567,  0.04575852, -0.18718385},
			 { 1.53277921,  1.46935877,  0.15494743,  0.37816252}});

	torch::Tensor omega_v = torch::tensor(
			{{-0.88778575, -1.98079647, -0.34791215,  0.15634897},
			{ 1.23029068,  1.20237985, -0.38732682, -0.30230275},
			 {-1.04855297, -1.42001794, -1.70627019,  1.9507754 },
			 {-0.50965218, -0.4380743,  -1.25279536,  0.77749036}});

	torch::Tensor beta_q = torch::tensor(
			{{-1.61389785},
			 {-0.21274028},
			 {-0.89546656},
			 { 0.3869025 }});

	torch::Tensor beta_k = torch::tensor(
			{{-0.51080514},
			 {-1.18063218},
			 {-0.02818223},
			 { 0.42833187}});

	torch::Tensor beta_v = torch::tensor(
			{{ 0.06651722},
			{ 0.3024719 },
			 {-0.63432209},
			 {-0.36274117}});

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute self attention in matrix form\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor all_x_prime = self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k);

	// Print out true values to check you have it correct
	std::cout << "x_prime_0_calculated: ";
	print(all_x_prime.t()[0].reshape({1, -1}));
	std::cout << '\n';
	printf("x_prime_0_true: [[ 0.94744244 -0.24348429 -0.91310441 -0.44522983]]\n");
	std::cout << "x_prime_1_calculated: ";
	print(all_x_prime.t()[1].reshape({1, -1}));
	std::cout << '\n';
	printf("x_prime_1_true: [[ 1.64201168 -0.08470004  4.02764044  2.18690791]]\n");
	std::cout << "x_prime_2_calculated: ";
	print(all_x_prime.t()[2].reshape({1, -1}));
	std::cout << '\n';
	printf("x_prime_2_true: [[ 1.61949281 -0.06641533  3.96863308  2.15858316]]\n");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute scaled dot product self-attention\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	all_x_prime = scaled_dot_product_self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k);
	print(all_x_prime.t());
	printf("\n");

	std::cout << "Done!\n";
	return 0;
}




