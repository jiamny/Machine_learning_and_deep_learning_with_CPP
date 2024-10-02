/*
 * C4_1_ComposingNeuralNetworks.cpp
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

// Define the Rectified Linear Unit (ReLU) function
torch::Tensor ReLU(torch::Tensor x) {

	if( x.dim() == 1)
		x = x.reshape({1, -1});

	return torch::where(x >= 0.0, x, 0.0);
}

//Define a shallow neural network with, one input, one output, and three hidden units
torch::Tensor
shallow_1_1_3(torch::Tensor x, torch::Tensor (*activation_fn)(torch::Tensor), double phi_0, double phi_1,double phi_2,
		double phi_3, double theta_10, double theta_11, double theta_20, double theta_21, double theta_30, double theta_31) {
	// Initial lines
	torch::Tensor pre_1 = theta_10 + theta_11 * x;
	torch::Tensor pre_2 = theta_20 + theta_21 * x;
	torch::Tensor pre_3 = theta_30 + theta_31 * x;

	// Activation functions
	torch::Tensor act_1 = activation_fn(pre_1);
	torch::Tensor act_2 = activation_fn(pre_2);
	torch::Tensor act_3 = activation_fn(pre_3);
	// Weight activations
	torch::Tensor w_act_1 = phi_1 * act_1;
	torch::Tensor w_act_2 = phi_2 * act_2;
	torch::Tensor w_act_3 = phi_3 * act_3;

	//Combine weighted activation and add y offset
	torch::Tensor y = phi_0 + w_act_1 + w_act_2 + w_act_3;
	// Return everything we have calculated
	return y;
}

// Plot two shallow neural networks and the composition of the two
void plot_neural_two_components(torch::Tensor x_in, torch::Tensor net1_out,
								torch::Tensor net2_out, torch::Tensor net12_out=torch::empty(0)) {

  // Plot the two networks separately
	matplot::figure(true)->size(800, 500);
  //fig, ax = plt.subplots(1,2)
  //fig.set_size_inches(8.5, 8.5)
  //fig.tight_layout(pad=3.0)
	matplot::subplot(1, 2, 0);
    matplot::plot(tensorTovector(x_in.to(torch::kDouble)),
    		tensorTovector(net1_out.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("Net 1 input");
    matplot::xlabel("Net 1 output");
    matplot::ylim({-1,1});
    matplot::xlim({-1,1});

    matplot::subplot(1, 2, 1);
    matplot::plot(tensorTovector(x_in.to(torch::kDouble)),
    		tensorTovector(net2_out.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("Net 2 input");
    matplot::xlabel("Net 2 output");
    matplot::ylim({-1,1});
    matplot::xlim({-1,1});
    matplot::show();

  if( net12_out.numel() > 0 ) {
	  // Plot their composition
	  matplot::figure(true)->size(700, 500);
	  matplot::plot(tensorTovector(x_in.to(torch::kDouble)),
			  tensorTovector(net12_out.to(torch::kDouble)), "g-")->line_width(2);
	  matplot::xlabel("Net 1 Input");
	  matplot::ylabel("Net 2 Output");
	  matplot::xlim({0,1});
	  matplot::ylim({-1,1});
	  matplot::show();
  }
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Now lets define some parameters and run the first neural network
	double n1_theta_10 = 0.0, n1_theta_11 = -1.0,
		   n1_theta_20 = 0, n1_theta_21 = 1.0,
		   n1_theta_30 = -0.67, n1_theta_31 =  1.0,
		   n1_phi_0 = 1.0, n1_phi_1 = -2.0, n1_phi_2 = -3.0, n1_phi_3 = 9.3;

	// Now lets define some parameters and run the second neural network
	double n2_theta_10 =  -0.6, n2_theta_11 = -1.0,
			n2_theta_20 =  0.2, n2_theta_21 = 1.0,
			n2_theta_30 =  -0.5, n2_theta_31 =  1.0,
			n2_phi_0 = 0.5, n2_phi_1 = -1.0, n2_phi_2 = -1.5, n2_phi_3 = 2.0;

	// Display the two inputs
	torch::Tensor x = torch::arange(-1,1,0.001);
	// We run the first  and second neural networks for each of these input values
	torch::Tensor net1_out = shallow_1_1_3(x, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10,
											n1_theta_11, n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);
	torch::Tensor net2_out = shallow_1_1_3(x, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10,
											n2_theta_11, n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);
	// Plot both graphs
	plot_neural_two_components(x, net1_out, net2_out);

	std::cout << "// -------------------------------------------------------------\n";
	std::cout << "// Feed the output of the first network into the second one.\n";
	std::cout << "// -------------------------------------------------------------\n";

	torch::Tensor net12_out = shallow_1_1_3(net1_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10,
												n2_theta_11, n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);

	plot_neural_two_components(x, net1_out, net2_out, net12_out);

	std::cout << "// -------------------------------------------------------------\n";
	std::cout << "// Change the second network? (note the *-1 change)\n";
	std::cout << "// -------------------------------------------------------------\n";
	net1_out = shallow_1_1_3(x, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11, n1_theta_20,
							n1_theta_21, n1_theta_30, n1_theta_31);
	net2_out = shallow_1_1_3(x, ReLU, n2_phi_0, n2_phi_1*-1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11, n2_theta_20,
							n2_theta_21, n2_theta_30, n2_theta_31);
	plot_neural_two_components(x, net1_out, net2_out);

	std::cout << "// -------------------------------------------------------------\n";
	std::cout << "// Feed the output of the first network into the second one.\n";
	std::cout << "// -------------------------------------------------------------\n";
	net12_out = shallow_1_1_3(net1_out, ReLU, n2_phi_0, n2_phi_1*-1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11,
							n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);
	plot_neural_two_components(x, net1_out, net2_out, net12_out);

	// Let's change things again.  What happens if we change the first network? (note the changes)
	net1_out = shallow_1_1_3(x, ReLU, n1_phi_0, n1_phi_1*0.5, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11,
							n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);
	net2_out = shallow_1_1_3(x, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11,
							n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);
	plot_neural_two_components(x, net1_out, net2_out);

	net12_out = shallow_1_1_3(net1_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11,
							n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);
	plot_neural_two_components(x, net1_out, net2_out, net12_out);

	// Let's change things again.  What happens if the first network and second networks are the same?
	net1_out = shallow_1_1_3(x, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11,
							n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);
	torch::Tensor net2_out_new = shallow_1_1_3(x, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11,
							n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);
	plot_neural_two_components(x, net1_out, net2_out_new);

	net12_out = shallow_1_1_3(net1_out, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11,
							n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31);
	plot_neural_two_components(x, net1_out, net2_out_new, net12_out);

	std::cout << "// -------------------------------------------------------------\n";
	std::cout << "// How many total linear regions will we have in the output?\n";
	std::cout << "// -------------------------------------------------------------\n";

	// How many total linear regions will we have in the output?
	torch::Tensor net123_out = shallow_1_1_3(net12_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11,
							n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31);
	plot_neural_two_components(x, net12_out, net2_out, net123_out);





	std::cout << "Done!\n";
	return 0;
}



