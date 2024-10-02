/*
 * C4_2_Clipping_functions.cpp
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
shallow_1_1_3_3(torch::Tensor x, torch::Tensor (*activation_fn)(torch::Tensor), torch::Tensor phi,
				torch::Tensor psi, torch::Tensor theta) {
	// Preactivations at layer 1 (terms in brackets in equation 4.7)
	torch::Tensor layer1_pre_1 = theta[1][0] + theta[1][1] * x;
	torch::Tensor layer1_pre_2 = theta[2][0] + theta[2][1] * x;
	torch::Tensor layer1_pre_3 = theta[3][0] + theta[3][1] * x;

	// Activation functions (rest of equation 4.7)
	torch::Tensor h1 = activation_fn(layer1_pre_1);
	torch::Tensor h2 = activation_fn(layer1_pre_2);
	torch::Tensor h3 = activation_fn(layer1_pre_3);

	// Preactivations at layer 2 (terms in brackets in equation 4.8)
	torch::Tensor layer2_pre_1 = psi[1][0] + psi[1][1] * h1 + psi[1][2] * h2 + psi[1][3] * h3;
	torch::Tensor layer2_pre_2 = psi[2][0] + psi[2][1] * h1 + psi[2][2] * h2 + psi[2][3] * h3;
	torch::Tensor layer2_pre_3 = psi[3][0] + psi[3][1] * h1 + psi[3][2] * h2 + psi[3][3] * h3;

	// Activation functions (rest of equation 4.8)
	torch::Tensor h1_prime = activation_fn(layer2_pre_1);
	torch::Tensor h2_prime = activation_fn(layer2_pre_2);
	torch::Tensor h3_prime = activation_fn(layer2_pre_3);

	// Weighted outputs by phi (three last terms of equation 4.9)
	torch::Tensor phi1_h1_prime = phi[1] * h1_prime;
	torch::Tensor phi2_h2_prime = phi[2] * h2_prime;
	torch::Tensor phi3_h3_prime = phi[3] * h3_prime;

	// Combine weighted activation and add y offset (summing terms of equation 4.9)
	torch::Tensor y = phi[0].data().item<double>() + phi1_h1_prime + phi2_h2_prime + phi3_h3_prime;


	// Return everything we have calculated
	return std::make_tuple(y, layer2_pre_1, layer2_pre_2, layer2_pre_3, h1_prime, h2_prime,
			  h3_prime, phi1_h1_prime, phi2_h2_prime, phi3_h3_prime);
}

void plot_neural_two_layers(torch::Tensor x, torch::Tensor y, torch::Tensor layer2_pre_1,torch::Tensor layer2_pre_2,
		torch::Tensor layer2_pre_3, torch::Tensor h1_prime, torch::Tensor h2_prime, torch::Tensor h3_prime,
		torch::Tensor phi1_h1_prime, torch::Tensor phi2_h2_prime, torch::Tensor phi3_h3_prime) {

	matplot::figure(true)->size(1000, 1000);
	matplot::subplot(3, 3, 0);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(layer2_pre_1.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("psi_10+psi_11h_1+psi_12h_2+psi_13h_3");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 1);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(layer2_pre_2.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("psi_20+psi_21h_1+psi_22h_2+psi_23h_3");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 2);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(layer2_pre_3.to(torch::kDouble)),"g-")->line_width(2);
    matplot::ylabel("psi_30+psi_31h_1+psi_32h_2+psi_33h_3");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 3);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(h1_prime.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("h_1'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 4);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(h2_prime.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("h_2'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 5);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(h3_prime.to(torch::kDouble)),"g-")->line_width(2);
    matplot::ylabel("h_3'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 6);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(phi1_h1_prime.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("phi_1 h_1'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 7);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(phi2_h2_prime.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("phi_2 h_2'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

	matplot::subplot(3, 3, 8);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(phi3_h3_prime.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("phi_3 h_3'");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});
    matplot::show();

	matplot::figure(true)->size(700, 500);
	matplot::plot(tensorTovector(x.to(torch::kDouble)),
				  tensorTovector(y.to(torch::kDouble)))->line_width(2);
	matplot::xlabel("Input, x");
	matplot::ylabel("Output, y");
	matplot::xlim({0,1});
	matplot::ylim({-1,1});
	matplot::show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Clipping functions\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor theta = torch::zeros({4,2}).to(torch::kDouble);
	torch::Tensor psi = torch::zeros({4,4}).to(torch::kDouble);
	torch::Tensor phi = torch::zeros({4,1}).to(torch::kDouble);

	theta[1][0] =  0.3; theta[1][1] = -1.0;
	theta[2][0]= -1.0; theta[2][1] = 2.0;
	theta[3][0] = -0.5; theta[3][1] = 0.65;
	psi[1][0] = 0.3;  psi[1][1] = 2.0; psi[1][2] = -1.0; psi[1][3]=7.0;
	psi[2][0] = -0.2;  psi[2][1] = 2.0; psi[2][2] = 1.2; psi[2][3]=-8.0;
	psi[3][0] = 0.3;  psi[3][1] = -2.3; psi[3][2] = -0.8; psi[3][3]=2.0;
	phi[0] = 0.0; phi[1] = 0.5; phi[2] = -1.5; phi [3] = 2.2;

	// Define a range of input values
	torch::Tensor x = torch::arange(0,1,0.01).to(torch::kDouble);

	// Run the neural network
	torch::Tensor y, layer2_pre_1, layer2_pre_2, layer2_pre_3, h1_prime,
	h2_prime, h3_prime, phi1_h1_prime, phi2_h2_prime, phi3_h3_prime;
	std::tie(y, layer2_pre_1, layer2_pre_2, layer2_pre_3, h1_prime, h2_prime, h3_prime,
			 phi1_h1_prime, phi2_h2_prime, phi3_h3_prime) = shallow_1_1_3_3(x, ReLU, phi, psi, theta);

	std::cout << y.sizes() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Plot two layer neural network as in figure 4.5\n";
	std::cout << "// --------------------------------------------------\n";
	plot_neural_two_layers(x, y, layer2_pre_1, layer2_pre_2, layer2_pre_3, h1_prime,
			h2_prime, h3_prime, phi1_h1_prime, phi2_h2_prime, phi3_h3_prime);

	std::cout << "Done!\n";
	return 0;
}




