/*
 * C4_3_Deep_Networks.cpp
 *
 *  Created on: Sep 20, 2024
 *      Author: jiamny
 */


#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor ReLU(torch::Tensor x) {
	if( x.dim() == 1)
		x = x.reshape({1, -1});

	return torch::where(x >=0.0, x, 0.0);
}

//Define a shallow neural network with, one input, one output, and three hidden units
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
shallow_1_1_3(torch::Tensor x, torch::Tensor (*activation_fn)(torch::Tensor), double phi_0, double phi_1, double phi_2,
		double phi_3, double theta_10, double theta_11, double theta_20, double theta_21, double theta_30, double theta_31) {
	// from the theta parameters (i.e. implement equations at bottom of figure 3.3a-c).  These are the preactivations
	torch::Tensor pre_1 = theta_10 + theta_11 * x;
	torch::Tensor pre_2 = theta_20 + theta_21 * x;
	torch::Tensor pre_3 = theta_30 + theta_31 * x;

	// Pass these through the ReLU function to compute the activations as in
	// figure 3.3 d-f
	torch::Tensor act_1 = activation_fn(pre_1);
	torch::Tensor act_2 = activation_fn(pre_2);
	torch::Tensor act_3 = activation_fn(pre_3);

	// weight the activations using phi1, phi2 and phi3
	// To create the equivalent of figure 3.3 g-i
	torch::Tensor w_act_1 = phi_1 * act_1;
	torch::Tensor w_act_2 = phi_2 * act_2;
	torch::Tensor w_act_3 = phi_3 * act_3;

	// combining the weighted activations and add
	// phi_0 to create the output as in figure 3.3 j
	torch::Tensor y = phi_0 + w_act_1 + w_act_2 + w_act_3;

	// Return everything we have calculated
  return std::make_tuple(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3);
}

void plot_neural(torch::Tensor x, torch::Tensor y) {
	matplot::figure(true)->size(700, 500);
	matplot::plot(tensorTovector(x.to(torch::kDouble)),
				  tensorTovector(y.to(torch::kDouble)))->line_width(2);
	matplot::xlabel("Input");
	matplot::ylabel("Output");
	matplot::xlim({-1,1});
	matplot::ylim({-1,1});
	matplot::show();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// -----------------------------------------------------\n";
	std::cout << "// General case of a deep network with two hidden layers\n";
	std::cout << "// -----------------------------------------------------\n";

	// Now lets define some parameters and run the first neural network
	double n1_theta_10 = 0.0, n1_theta_11 = -1.0,
	n1_theta_20 = 0, n1_theta_21 = 1.0,
	n1_theta_30 = -0.67, n1_theta_31 =  1.0,
	n1_phi_0 = 1.0, n1_phi_1 = -2.0, n1_phi_2 = -3.0, n1_phi_3 = 9.3;

	// Define a range of input values
	torch::Tensor n1_in = torch::arange(-1,1,0.01).to(torch::kDouble).reshape({1,-1});

	// We run the neural network for each of these input values
	torch::Tensor y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3;
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
		shallow_1_1_3(n1_in, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11,
						n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);

	// And then plot it
	plot_neural(n1_in, y);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Describe neural network in matrix notation\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor beta_0 = torch::zeros({3,1}).to(torch::kDouble);
	torch::Tensor Omega_0 = torch::zeros({3,1}).to(torch::kDouble);
	torch::Tensor beta_1 = torch::zeros({1,1}).to(torch::kDouble);
	torch::Tensor Omega_1 = torch::zeros({1,3}).to(torch::kDouble);

	beta_0[0][0] = n1_theta_10;
	beta_0[1][0] = n1_theta_20;
	beta_0[2][0] = n1_theta_30;

	Omega_0[0][0] = n1_theta_11;
	Omega_0[1][0] = n1_theta_21;
	Omega_0[2][0] = n1_theta_31;

	beta_1[0][0]  = n1_phi_0;
	Omega_1[0][0] = n1_phi_1;
	Omega_1[0][1] = n1_phi_2;
	Omega_1[0][2] = n1_phi_3;

	// Make sure that input data matrix has different inputs in its columns
	int n_data = n1_in.size(1);
	int n_dim_in = 1;
	torch::Tensor n1_in_mat = torch::reshape(n1_in, {n_dim_in, n_data});
	std::cout << n1_in_mat.sizes() << '\n';

	// This runs the network for ALL of the inputs, x at once so we can draw graph
	torch::Tensor h1 = ReLU(torch::matmul(beta_0, torch::ones({1, n_data}).to(torch::kDouble)) +
							torch::matmul(Omega_0, n1_in_mat));
	std::cout << h1.sizes() << '\n';
	torch::Tensor n1_out = torch::matmul(beta_1, torch::ones({1, n_data}).to(torch::kDouble)) +
							torch::matmul(Omega_1, h1);

	// Draw the network and check that it looks the same as the non-matrix case
	plot_neural(n1_in, n1_out);

	std::cout << "// -------------------------------------------------------------------\n";
	std::cout << "// Run the second neural network on the output of the first network\n";
	std::cout << "// -------------------------------------------------------------------\n";

	double n2_theta_10 =  -0.6, n2_theta_11 = -1.0,
	n2_theta_20 =  0.2, n2_theta_21 = 1.0,
	n2_theta_30 =  -0.5, n2_theta_31 =  1.0,
	n2_phi_0 = 0.5, n2_phi_1 = -1.0, n2_phi_2 = -1.5, n2_phi_3 = 2.0;

	// Define a range of input values
	torch::Tensor n2_in = torch::arange(-1,1,0.01).to(torch::kDouble);
	torch::Tensor n2_out;
	std::tie(n2_out, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
			shallow_1_1_3(n1_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11, n2_theta_20,
						n2_theta_21, n2_theta_30, n2_theta_31);

	plot_neural(n1_in, n2_out);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// General formulation\n";
	std::cout << "// --------------------------------------------------\n";
	beta_0 = torch::zeros({3,1}).to(torch::kDouble);
	Omega_0 = torch::zeros({3,1}).to(torch::kDouble);
	beta_1 = torch::zeros({3,1}).to(torch::kDouble);
	Omega_1 = torch::zeros({3,3}).to(torch::kDouble);
	torch::Tensor beta_2 = torch::zeros({1,1}).to(torch::kDouble);
	torch::Tensor Omega_2 = torch::normal(0., 1.0, {1, 3}).to(torch::kDouble); //torch::ones({1,3}).to(torch::kDouble);

	beta_0[0][0] = n1_theta_10;
	beta_0[1][0] = n1_theta_20;
	beta_0[2][0] = n1_theta_30;

	Omega_0[0][0] = n1_theta_11;
	Omega_0[1][0] = n1_theta_21;
	Omega_0[2][0] = n1_theta_31;

	beta_1[0][0] = n2_theta_10 + n2_theta_11 * n1_phi_0;
	beta_1[1][0] = n2_theta_20 + n2_theta_21 * n1_phi_0;
	beta_1[2][0] = n2_theta_30 + n2_theta_31 * n1_phi_0;

	Omega_1[0][0] = n2_theta_11 * n1_phi_1;
	Omega_1[0][1] = n2_theta_11 * n1_phi_2;
	Omega_1[0][2] = n2_theta_11 * n1_phi_3;
	Omega_1[1][0] = n2_theta_21 * n1_phi_1;
	Omega_1[1][1] = n2_theta_21 * n1_phi_2;
	Omega_1[1][2] = n2_theta_21 * n1_phi_3;
	Omega_1[2][0] = n2_theta_31 * n1_phi_1;
	Omega_1[2][1] = n2_theta_31 * n1_phi_2;
	Omega_1[2][2] = n2_theta_31 * n1_phi_3;


	n1_in_mat = torch::reshape(n1_in,{n_dim_in,n_data});

	std::cout << torch::matmul(beta_0, torch::ones({1,n_data}).to(torch::kDouble)).sizes() << '\n';

	// Make sure that input data matrix has different inputs in its columns
	h1 = ReLU(torch::matmul(beta_0, torch::ones({1,n_data}).to(torch::kDouble)) + torch::matmul(Omega_0,n1_in_mat));
	std::cout << h1.sizes() << '\n';
	torch::Tensor h2 = ReLU(torch::matmul(beta_1, torch::ones({1,n_data}).to(torch::kDouble)) + torch::matmul(Omega_1,h1));
	std::cout << h2 << '\n';
	std::cout << Omega_2 << '\n';
	std::cout << torch::matmul(beta_2, torch::ones({1, n_data}).to(torch::kDouble)) << '\n';
	n1_out = torch::matmul(beta_2, torch::ones({1, n_data}).to(torch::kDouble)) + torch::matmul(Omega_2, h2);
	std::cout << n1_out << '\n';

	// Draw the network and check that it looks the same as the non-matrix version
	plot_neural(n1_in, n1_out);

	std::cout << "Done!\n";
	return 0;
}



