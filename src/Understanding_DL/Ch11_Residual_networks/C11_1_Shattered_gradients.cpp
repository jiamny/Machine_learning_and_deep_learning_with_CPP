/*
 * C11_1_Shattered_gradients.cpp
 *
 *  Created on: Feb 8, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/UDL_util.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor ReLU(torch::Tensor x) {

	if( x.dim() == 1)
		x = x.reshape({1, -1});

	return torch::where(x >= 0.0, x, 0.0);
}

// K is width, D is number of hidden units in each layer
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> init_params(int K, int D) {
	// Input layer
	int D_i = 1;
	// Output layer
	int D_o = 1;

	// Glorot initialization
	float sigma_sq_omega = 1.0/D;

	// Make empty lists
	std::vector<torch::Tensor> all_weights(K+1);
	std::vector<torch::Tensor> all_biases(K+1);

	// Create parameters for input and output layers
	all_weights[0] = torch::randn({D, D_i}) * std::sqrt(sigma_sq_omega);
	all_weights[K] = torch::randn({D_o, D}) * std::sqrt(sigma_sq_omega);
	all_biases[0] = torch::randn({D,1}) * std::sqrt(sigma_sq_omega);
	all_biases[K]= torch::randn({D_o,1}) * std::sqrt(sigma_sq_omega);

	// Create intermediate layers
	for(auto& layer : range(K-1, 1)) {
	  all_weights[layer] = torch::randn({D,D}) * std::sqrt(sigma_sq_omega);
	  all_biases[layer] = torch::randn({D,1}) * std::sqrt(sigma_sq_omega);
	}

	return std::make_tuple(all_weights, all_biases);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward_pass(
		torch::Tensor net_input, std::vector<torch::Tensor> all_weights, std::vector<torch::Tensor> all_biases) {
	// Retrieve number of layers
	  int K = all_weights.size() - 1;

	  // We'll store the pre-activations at each layer in a list "all_f"
	  // and the activations in a second list[all_h].
	  std::vector<torch::Tensor> all_f(K+1);
	  std::vector<torch::Tensor> all_h(K+1);

	  //For convenience, we'll set
	  // all_h[0] to be the input, and all_f[K] will be the output
	  all_h[0] = net_input;

	  // Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
	  for(auto& layer : range(K, 0)) {
	      // Update preactivations and activations at this layer according to eqn 7.5
	      all_f[layer] = all_biases[layer] + torch::matmul(all_weights[layer], all_h[layer]);
	      all_h[layer+1] = ReLU(all_f[layer]);
	  }

	  // Compute the output from the last hidden layer
	  all_f[K] = all_biases[K] + torch::matmul(all_weights[K], all_h[K]);

	  // Retrieve the output
	  torch::Tensor net_output = all_f[K];

	  return std::make_tuple(net_output, all_f, all_h);
}

// We'll need the indicator function
torch::Tensor indicator_function(torch::Tensor x) {
	torch::Tensor x_in = x.clone();
	x_in.masked_fill_(x_in.greater_equal(0), 1);
	x_in.masked_fill_(x_in.less(0), 0);
  return x_in;
}

//  Main backward pass routine
std::tuple<torch::Tensor, torch::Tensor> calc_input_output_gradient(torch::Tensor x_in,
		std::vector<torch::Tensor> all_weights, std::vector<torch::Tensor> all_biases) {

	int K = all_weights.size() - 1;
	//   Run the forward pass
	torch::Tensor y;
	std::vector<torch::Tensor> all_f, all_h;
	std::tie(y, all_f, all_h ) = forward_pass(x_in, all_weights, all_biases);

	// We'll store the derivatives dl_dweights and dl_dbiases in lists as well
	std::vector<torch::Tensor> all_dl_dweights(K+1);
	std::vector<torch::Tensor>  all_dl_dbiases(K+1);
	// And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
	std::vector<torch::Tensor> all_dl_df(K+1);
	std::vector<torch::Tensor> all_dl_dh(K+1);
	// Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

	// Compute derivatives of net output with respect to loss
	all_dl_df[K] = torch::ones_like(all_f[K]);

	// Now work backwards through the network
	for(auto& i : range(K+1,0)) {
		int layer = K - i;
	    all_dl_dbiases[layer] = all_dl_df[layer].clone();
	    all_dl_dweights[layer] = torch::matmul(all_dl_df[layer], all_h[layer].t());

	    all_dl_dh[layer] = torch::matmul(all_weights[layer].t(), all_dl_df[layer]);

	    if( layer > 0)
	      all_dl_df[layer-1] = indicator_function(all_f[layer-1]).multiply(all_dl_dh[layer]);
	}

	return std::make_tuple(all_dl_dh[0],  y);
}

void plot_derivatives(int K, int D, std::string tlt="") {

	// Initialize parameters
	std::vector<torch::Tensor> all_weights, all_biases;
	std::tie(all_weights, all_biases) = init_params(K,D);

	torch::Tensor x_in = torch::arange(-2, 2, 4.0/256.0);
	x_in = torch::resize(x_in, {1, x_in.size(0)});
	torch::Tensor dydx, y;
	std::tie(dydx, y) = calc_input_output_gradient(x_in, all_weights, all_biases);

	matplot::figure(true)->size(800, 600);
	matplot::plot(tensorTovector(x_in.squeeze().to(torch::kDouble)),
			  tensorTovector(dydx.squeeze().to(torch::kDouble)))->line_width(2);
	matplot::xlabel("Input, x");
	matplot::ylabel("Gradient, dy/dx");
	matplot::xlim({-2,2});

	if(tlt != "")
		matplot::title(tlt);

	matplot::show();
}

void plot_autocorr(int K, int D) {

	// Initialize parameters
	std::vector<torch::Tensor> all_weights, all_biases;
	std::tie(all_weights, all_biases) = init_params(K,D);

	torch::Tensor x_in = torch::arange(-2, 2, 4.0/256.0);
	x_in = torch::resize(x_in, {1, x_in.size(0)});
	torch::Tensor dydx, y;
	std::tie(dydx, y) = calc_input_output_gradient(x_in, all_weights, all_biases);

	std::vector<double> a = tensorTovector(dydx.squeeze().to(torch::kDouble));
	std::vector<double> v = tensorTovector(dydx.squeeze().clone().to(torch::kDouble));

	std::string mode = "same";
	std::vector<double> ac = linear_cross_correlation(a, v, mode);
	double dd = ac[128];
	for(auto& i : range(static_cast<int>(ac.size()), 0)) {
		ac[i] /= dd;
	}

	std::vector<double> yy(ac.begin()+128, ac.end());
	std::vector<double> vxin = tensorTovector(x_in.squeeze().to(torch::kDouble));
	std::vector<double> xx(vxin.begin()+128, vxin.end());

	matplot::figure(true)->size(800, 600);
	matplot::plot(xx, yy)->line_width(2);
	matplot::xlabel("Distance");
	matplot::ylabel("Autocorrelation");
	matplot::xlim({0,2});
	matplot::title( "Neurons = " + std::to_string(D) + ", layers = " + std::to_string(K));

	matplot::show();

}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

	int D = 200, K = 3;
	// Initialize parameters
	std::vector<torch::Tensor>  all_weights, all_biases;
	std::tie(all_weights, all_biases) = init_params(K, D);
	std::cout << all_weights[0].sizes() << '\n';
	torch::Tensor x = torch::ones({1,1});
	torch::Tensor dydx, y;
	std::tie(dydx, y) = calc_input_output_gradient(x, all_weights, all_biases);

	// Offset for finite gradients
	float delta = 0.001;
	torch::Tensor x1 = x.clone(), y1, x2, y2;
	std::vector<torch::Tensor> t1, t2;
	std::tie(y1, t1, t2) = forward_pass(x1, all_weights, all_biases);

	x2 = x+delta;
	std::tie(y2, t1, t2) = forward_pass(x2, all_weights, all_biases);

	// Finite difference calculation
	torch::Tensor dydx_fd = (y2-y1)/delta;

	printf("Gradient calculation=%f, Finite difference gradient=%f\n",
			dydx.squeeze().data().item<float>(), dydx_fd.squeeze().data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Build a model with one hidden layer and 200 neurons and plot derivatives\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	D = 200;
	K = 1;
	if(plt) plot_derivatives(K, D, std::to_string(K)  + " hidden layer");

	std::cout << "// Build a model with 24 hidden layer and 200 neurons and plot derivatives\n";
	K = 24;
	if(plt) plot_derivatives(K, D, std::to_string(K)  + " hidden layer");

	std::cout << "// Build a model with 50 hidden layer and 200 neurons and plot derivatives\n";
	K = 50;
	if(plt) plot_derivatives(K, D, std::to_string(K)  + " hidden layer");

	std::cout << "// Build a model with 52 hidden layer and 200 neurons and plot derivatives\n";
	K = 52;
	if(plt) plot_derivatives(K, D, std::to_string(K)  + " hidden layer");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Autocorrelation\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// D=200, K=1\n";
	K = 1;
	D = 200;
	plot_autocorr(K, D);

	std::cout << "// D=200, K=50\n";
	K = 50;
	plot_autocorr(K, D);

	std::cout << "// D=10, K=1\n";
	K = 1;
	D = 10;
	plot_autocorr(K, D);

	std::cout << "// D=10, K=5\n";
	K = 5;
	plot_autocorr(K, D);

//	std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
//	std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};
//    printVector(v);
//    printVector(a);
//	std::string mode = "same";
//	std::vector<double> c = linear_cross_correlation(a, v, mode);
//	printVector(c);

	std::cout << "Done!\n";
	return 0;
}



