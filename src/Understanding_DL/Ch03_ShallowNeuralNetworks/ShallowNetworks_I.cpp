/*
 * ShallowNetworks_I.cpp
 *
 *  Created on: Sep 1, 2024
 *      Author: hhj
 */


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

torch::Tensor least_squares_loss(torch::Tensor y_train, torch::Tensor y_predict) {
	// compute the sum of squared
	// differences between the real values of y and the predicted values from the model f[x_i,phi]
	// (see figure 2.2 of the book)
	torch::Tensor loss = torch::sum(torch::pow(y_predict - y_train, 2));

	return loss;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Make an array of inputs
	torch::Tensor z = torch::arange(-5,5,0.1);
	torch::Tensor RelU_z = ReLU(z);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Plot the ReLU function\n";
	std::cout << "// --------------------------------------------------\n";

	matplot::plot(tensorTovector(z.to(torch::kDouble)),
					tensorTovector(RelU_z.to(torch::kDouble)), "r-")->line_width(2);
	matplot::xlim({-5,5});
	matplot::ylim({-5,5});
	matplot::xlabel("z");
	matplot::ylabel("ReLU[z]");
	matplot::show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Now lets define some parameters and run the neural network\n";
	std::cout << "// --------------------------------------------------\n";
/*
	double theta_10 =  -4.0,  theta_11 = 0.9, theta_12 = 0.0,
	theta_20 =  5.0, theta_21 = -0.9,  theta_22 = -0.5,
	theta_30 =  -7, theta_31 = 0.5, theta_32 = 0.9,
	phi_0 = 0.0, phi_1 = -2.0, phi_2 = 2.0, phi_3 = 1.5;

	torch::Tensor x1 = torch::arange(0.0, 10.0, 0.1);
	torch::Tensor x2 = torch::arange(0.0, 10.0, 0.1);
	auto t = torch::meshgrid({x1,x2});  // https://www.geeksforgeeks.org/numpy-meshgrid-function/
	x1 = t[0];
	x2 = t[1];
*/
	double theta_10 = 0.3, theta_11 = -1.0,
	theta_20 = -1.0, theta_21 = 2.0,
	theta_30 = -0.5, theta_31 = 0.65,
	phi_0 = -0.3, phi_1 = 2.0, phi_2 = -1.0, phi_3 = 7.0;

	// Define a range of input values
	torch::Tensor x = torch::arange(0,1,0.01);
	// We run the neural network for each of these input values
	torch::Tensor y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3;

	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
			shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);

	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Least squares function\n";
	std::cout << "// --------------------------------------------------\n";

	// Now lets define some parameters, run the neural network, and compute the loss
	theta_10 =  0.3; theta_11 = -1.0;
	theta_20 = -1.0; theta_21 = 2.0;
	theta_30 = -0.5; theta_31 = 0.65;
	phi_0 = -0.3; phi_1 = 2.0; phi_2 = -1.0; phi_3 = 7.0;

	// Define a range of input values
	x = torch::arange(0, 1, 0.01);

	torch::Tensor x_train = torch::tensor({0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,
	                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,
	                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,
	                   0.87168699,0.58858043});
	torch::Tensor y_train = torch::tensor({-0.15934537,0.18195445,0.451270150,0.13921448,0.09366691,0.30567674,
	                    0.372291170,0.40716968,-0.08131792,0.41187806,0.36943738,0.3994327,
	                    0.019062570,0.35820410,0.452564960,-0.0183121,0.02957665,-0.24354444,
	                    0.148038840,0.26824970});

	// We run the neural network for each of these input values
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
	    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31);
	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true, x_train, y_train);

	// Run the neural network on the training data
	torch::Tensor y_predict = std::get<0>( shallow_1_1_3(x_train, ReLU, phi_0,phi_1,phi_2, phi_3,
			theta_10, theta_11, theta_20, theta_21, theta_30, theta_31));

	// Compute the least squares loss and print it out
	torch::Tensor loss = least_squares_loss(y_train, y_predict);
	printf("Your Loss = %3.3f, True value = 9.385\n", loss.data().item<double>());

	// Manipulate the parameters (by hand!) to make the function
	// fit the data better and try to reduce the loss to as small a number
	// as possible.  The best that I could do was 0.181

	double best_L = 1000.0, b_phi_0 = 0., b_phi_1 = 0., b_phi_2 = 0., b_phi_3 = 0.;

	for(double phi_0 = -1.0; phi_0 <= 1.0; phi_0 += 0.01) {
		for(double phi_1 = -10.0; phi_1 <= 10.0; phi_1 += 1.0) {
			for(double phi_2 = -10.0; phi_2 <= 10.0; phi_2 += 1.0) {
				for(double phi_3 = -10.0; phi_3 <= 10.0; phi_3 += 1.0) {
					y_predict = std::get<0>( shallow_1_1_3(x_train, ReLU, phi_0,phi_1,phi_2, phi_3,
										theta_10, theta_11, theta_20, theta_21, theta_30, theta_31));

					double n_L = least_squares_loss(y_train, y_predict).data().item<double>();
					if(n_L < best_L) {
						best_L = n_L;
						b_phi_0 = phi_0;
						b_phi_1 = phi_1;
						b_phi_2 = phi_2;
						b_phi_3 = phi_3;
					}
				}
			}
		}
	}
	printf("By manipulating the phi parameters, the best that I could do is: %3.3f; "
			"phi_0 = %.4f, phi_1 = %.4f, phi_2 = %.4f, phi_3 = %.4f.\n",
			best_L, b_phi_0, b_phi_1, b_phi_2, b_phi_3);

	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
		    shallow_1_1_3(x, ReLU, b_phi_0, b_phi_1, b_phi_2, b_phi_3, theta_10, theta_11,
		    		theta_20, theta_21, theta_30, theta_31);
	// And then plot it
	plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, true, x_train, y_train);

	std::cout << "Done!\n";
	return 0;
}


