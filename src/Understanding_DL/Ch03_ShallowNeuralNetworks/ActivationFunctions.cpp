/*
 * ActivationFunctions.cpp
 *
 *  Created on: Sep 12, 2024
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

	return torch::where(x >=0.0, x, 0.0);
}

torch::Tensor heaviside(torch::Tensor z) {
    if( z.dim() == 1)
    	z = z.reshape({1, -1});

    //z = torch::where(z < 0.0, 0.0, 1.0);
    z = torch::where(z > 0.0, 1.0, z);
    z = torch::where(z == 0.0, 0.5, z);
    z = torch::where(z < 0.0, 0.0, z);
    return z;
}

torch::Tensor lin( torch::Tensor preactivation) {
	// Try a=0.5, b=-0.4 Don't forget to run the cell again to update the function
	// double a = 0., b = 1.0;
	double a = 0.5, b = 0.4;

	// Compute linear function
	torch::Tensor activation = a + b * preactivation;
	// Return
	return activation;
}

// Plot the shallow neural network.  We'll assume input in is range [0,1] and output [-1,1]
// If the plot_all flag is set to true, then we'll plot all the intermediate stages as in Figure 3.3
void plot_neural(torch::Tensor x, torch::Tensor y, torch::Tensor pre_1, torch::Tensor pre_2, torch::Tensor pre_3,
		torch::Tensor act_1, torch::Tensor act_2, torch::Tensor act_3, torch::Tensor w_act_1, torch::Tensor w_act_2,
		torch::Tensor w_act_3, bool plot_all=false, torch::Tensor x_data=torch::empty(0), torch::Tensor y_data=torch::empty(0)) {

  matplot::figure(true)->size(1000, 1000);
  // Plot intermediate plots if flag set
  if(plot_all) {

	matplot::subplot(3, 3, 0);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(pre_1.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("Preactivation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 1);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(pre_2.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("Preactivation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 2);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(pre_3.to(torch::kDouble)),"g-")->line_width(2);
    matplot::ylabel("Preactivation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 3);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(act_1.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("Activation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 4);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(act_2.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("Activation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 5);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(act_3.to(torch::kDouble)),"g-")->line_width(2);
    matplot::ylabel("Activation");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 6);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(w_act_1.to(torch::kDouble)),"r-")->line_width(2);
    matplot::ylabel("Weighted Act");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 7);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(w_act_2.to(torch::kDouble)),"b-")->line_width(2);
    matplot::ylabel("Weighted Act");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::subplot(3, 3, 8);
    matplot::plot(tensorTovector(x.to(torch::kDouble)),
    		tensorTovector(w_act_3.to(torch::kDouble)),"g-")->line_width(2);
    matplot::ylabel("Weighted Act");
    matplot::xlabel("Input, x");
    matplot::ylim({-1,1});
    matplot::xlim({0,1});

    matplot::show();
  }

  matplot::figure(true)->size(700, 500);
  matplot::plot(tensorTovector(x.to(torch::kDouble)),
		  tensorTovector(y.to(torch::kDouble)))->line_width(2);
  matplot::xlabel("Input, x");
  matplot::ylabel("Output, y");
  matplot::xlim({0,1});
  matplot::ylim({-1,1});

  if(x_data.numel() > 0) {
      matplot::hold(true);
	  matplot::plot(tensorTovector(x_data.to(torch::kDouble)),
			  tensorTovector(y_data.to(torch::kDouble)), "mo")->line_width(2);
	  for(auto& i : range(static_cast<int>(x_data.size(0)), 0))
		  matplot::plot(tensorTovector(x_data[i].to(torch::kDouble)),
				  tensorTovector(y_data[i].to(torch::kDouble)))->line_width(2);
  }
  matplot::show();
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



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// ReLU\n";
	std::cout << "// --------------------------------------------------\n";

	// Now lets define some parameters and run the neural network
	double theta_10 =  0.3, theta_11 = -1.0,
		   theta_20 = -1.0, theta_21 = 2.0,
		   theta_30 = -0.5, theta_31 = 0.65,
		   phi_0 = -0.3, phi_1 = 2.0, phi_2 = -1.0, phi_3 = 7.0;

	// Define a range of input values
	torch::Tensor x = torch::arange(0,1,0.01).to(torch::kDouble);

	// We run the neural network for each of these input values
	torch::Tensor y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3;
    std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
    		shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);
    // And then plot it
    plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Sigmoid activation functio\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor z = torch::arange(-1.0, 1.0, 0.01).to(torch::kDouble);
	torch::Tensor sig_z = Sigmoid(z);

	// Plot the sigmoid function
	matplot::plot(tensorTovector(z), tensorTovector(sig_z), "r-")->line_width(2);
	matplot::xlim({-1,1});
	matplot::ylim({0,1});
	matplot::xlabel("z");
	matplot::ylabel("sig[z]");
	matplot::show();

	theta_10 =  0.3; theta_11 = -1.0;
	theta_20 = -1.0; theta_21 = 2.0;
	theta_30 = -0.5; theta_31 = 0.65;
	phi_0 = 0.3; phi_1 = 0.5; phi_2 = -1.0; phi_3 = 0.9;

	// Define a range of input values
	x = torch::arange(0.,1.,0.01).to(torch::kDouble);

	// We run the neural network for each of these input values
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
	    shallow_1_1_3(x, Sigmoid, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);
	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true);


	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Heaviside activation function\n";
	std::cout << "// --------------------------------------------------\n";
	// Make an array of inputs
	z = torch::arange(-1.0, 1.0, 0.01).to(torch::kDouble);
	torch::Tensor heav_z = heaviside(z);

	// Plot the heaviside function
	matplot::plot(tensorTovector(z), tensorTovector(heav_z), "r-")->line_width(2);
	matplot::xlim({-1,1});
	matplot::ylim({-2,2});
	matplot::xlabel("z");
	matplot::ylabel("heaviside[z]");
	matplot::show();

	theta_10 =  0.3; theta_11 = -1.0;
	theta_20 = -1.0; theta_21 = 2.0;
	theta_30 = -0.5; theta_31 = 0.65;
	phi_0 = 0.3; phi_1 = 0.5; phi_2 = -1.0; phi_3 = 0.9;

	// Define a range of input values
	x = torch::arange(0.,1.,0.01).to(torch::kDouble);

	// We run the neural network for each of these input values
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
	    shallow_1_1_3(x, heaviside, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);

	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Linear activation functions\n";
	std::cout << "// --------------------------------------------------\n";

	theta_10 =  0.3; theta_11 = -1.0;
	theta_20 = -1.0; theta_21 = 2.0;
	theta_30 = -0.5; theta_31 = 0.65;
	phi_0 = 0.3; phi_1 = 0.5; phi_2 = -1.0; phi_3 = 0.9;

	// Define a range of input values
	x = torch::arange(0.,1.,0.01).to(torch::kDouble);

	// We run the neural network for each of these input values
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
	    shallow_1_1_3(x, lin, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);
	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true);


	std::cout << "Done!\n";
	return 0;
}

