/*
 * ShallowNetworks_2.cpp
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

//Define a shallow neural network with, one input, one output, and three hidden units
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
shallow_2_1_3(torch::Tensor x1, torch::Tensor x2, torch::Tensor (*activation_fn)(torch::Tensor), double phi_0, double phi_1,
		double phi_2, double phi_3, double theta_10, double theta_11, double theta_12,
		double theta_20, double theta_21, double theta_22, double theta_30, double theta_31, double theta_32) {
	// from the theta parameters (i.e. implement equations at bottom of figure 3.3a-c).  These are the preactivations
	torch::Tensor pre_1 = theta_10 + theta_11 * x1 + theta_12 * x2;
	torch::Tensor pre_2 = theta_20 + theta_21 * x1 + theta_22 * x2;
	torch::Tensor pre_3 = theta_30 + theta_31 * x1 + theta_32 * x2;;

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

void draw_2D_function(matplot::axes_handle ax, torch::Tensor x1,  torch::Tensor x2, torch::Tensor y, std::string tlt) {

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < x1.size(0);  i += 1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (int j = 0; j < x1.size(1); j += 1) {
	            x_row.push_back(x1[i][j].data().item<double>());
	            y_row.push_back(x2[i][j].data().item<double>());
	            z_row.push_back(y[i][j].data().item<double>());
	    }
	    X.push_back(x_row);
	    Y.push_back(y_row);
	    Z.push_back(z_row);
	}

	std::vector<double> lvls = linspace(-10.0, 10.0, 20);

	matplot::contour(ax, X, Y, Z)->line_width(2).levels(lvls);
	matplot::xlabel(ax, "X1");
	matplot::ylabel(ax, "X2");
	matplot::title(ax, tlt);
}

// Plot the shallow neural network.  We'll assume input in is range [0,10],[0,10] and output [-10,10]
void plot_neural_2_inputs(torch::Tensor x1, torch::Tensor x2, torch::Tensor y, torch::Tensor pre_1, torch::Tensor pre_2,
		torch::Tensor pre_3, torch::Tensor act_1, torch::Tensor act_2, torch::Tensor act_3,
		torch::Tensor w_act_1, torch::Tensor w_act_2, torch::Tensor w_act_3) {

	auto F = figure(true);
	F->size(1000, 1000);
	F->reactive_mode(false);
	F->tiledlayout(3, 3);
	F->position(0, 0);

	matplot::axes_handle ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_1, "Preactivation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_2, "Preactivation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_3, "Preactivation");

	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_1, "Activation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_2, "Activation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_3, "Activation");

	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_1, "Weighted Act");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_2, "Weighted Act");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_3, "Weighted Act");
	F->draw();
	matplot::show();


	auto F2 = figure(true);
	F2->size(800, 600);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);

	matplot::axes_handle ax2 = F2->nexttile();
	draw_2D_function(ax2, x1, x2, y, "Network output, y");
	F2->draw();
	matplot::show();
}

// Plot the shallow neural network.  We'll assume input in is range [0,10],[0,10] and output [-10,10]
void plot_neural_2_inputs_2_outputs(torch::Tensor x1, torch::Tensor x2, torch::Tensor y1, torch::Tensor y2, torch::Tensor pre_1,
		torch::Tensor pre_2, torch::Tensor pre_3, torch::Tensor act_1, torch::Tensor act_2, torch::Tensor act_3, torch::Tensor w_act_11,
		torch::Tensor w_act_12, torch::Tensor w_act_13, torch::Tensor w_act_21, torch::Tensor w_act_22, torch::Tensor w_act_23) {

    // Plot intermediate plots if flag set
	auto F = figure(true);
	F->size(1200, 1000);
	F->reactive_mode(false);
	F->tiledlayout(4, 3);
	F->position(0, 0);

	matplot::axes_handle ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_1, "Preactivation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_2, "Preactivation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, pre_3, "Preactivation");

	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_1, "Activation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_2, "Activation");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, act_3, "Activation");

	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_11, "Weighted Act 1");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_12, "Weighted Act 1");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_13, "Weighted Act 1");

	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_21, "Weighted Act 2");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_22, "Weighted Act 2");
	ax = F->nexttile();
	draw_2D_function(ax, x1, x2, w_act_23, "Weighted Act 2");

	F->draw();
	matplot::show();

	auto F2 = figure(true);
	F2->size(800, 600);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);

	matplot::axes_handle ax2 = F2->nexttile();
	draw_2D_function(ax2, x1, x2, y1, "Network output, y_1");
	F2->draw();
	matplot::show();

	auto F3 = figure(true);
	F3->size(800, 600);
	F3->add_axes(false);
	F3->reactive_mode(false);
	F3->tiledlayout(1, 1);
	F3->position(0, 0);

	matplot::axes_handle ax3 = F3->nexttile();
	draw_2D_function(ax3, x1, x2, y2, "Network output, y_2");
	F3->draw();
	matplot::show();
}

// Define a shallow neural network with, two inputs, two outputs, and three hidden units
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
shallow_2_2_3(torch::Tensor x1, torch::Tensor x2, torch::Tensor (*activation_fn)(torch::Tensor), double phi_10, double phi_11, double phi_12, double phi_13,
		double phi_20, double phi_21, double phi_22, double phi_23, double theta_10, double theta_11,
        double theta_12, double theta_20, double theta_21, double  theta_22, double theta_30, double theta_31, double theta_32) {

    // replace the dummy code below
	torch::Tensor pre_1 = theta_10 + theta_11 * x1 + theta_12 * x2;
	torch::Tensor pre_2 = theta_20 + theta_21 * x1 + theta_22 * x2;
	torch::Tensor pre_3 = theta_30 + theta_31 * x1 + theta_32 * x2;

	torch::Tensor act_1 = activation_fn(pre_1);
	torch::Tensor act_2 = activation_fn(pre_2);
	torch::Tensor act_3 = activation_fn(pre_3);

	// weight the activations using phi_x
	torch::Tensor w_act_11 = phi_11 * act_1;
	torch::Tensor w_act_12 = phi_12 * act_2;
	torch::Tensor w_act_13 = phi_13 * act_3;

	torch::Tensor w_act_21 = phi_21 * act_1;
	torch::Tensor w_act_22 = phi_22 * act_2;
	torch::Tensor w_act_23 = phi_23 * act_3;

	// combining the weighted activations and add
	torch::Tensor y1 = phi_10 + w_act_11 + w_act_12 + w_act_13;
	torch::Tensor y2 = phi_20 + w_act_21 + w_act_22 + w_act_23;

  // Return everything we have calculated
  return std::make_tuple(y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12,
		  w_act_13, w_act_21, w_act_22, w_act_23);
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// ----------------------------------------------------------------------------\n";
	std::cout << "// Shallow neural network with, two input, one output, and three hidden units\n";
	std::cout << "// ----------------------------------------------------------------------------\n";
	// Now lets define some parameters and run the neural network
	double theta_10 =  -4.0,  theta_11 = 0.9, theta_12 = 0.0,
		   theta_20 =  5.0, theta_21 = -0.9, theta_22 = -0.5,
		   theta_30 =  -7.0, theta_31 = 0.5, theta_32 = 0.9,
		   phi_0 = 0.0, phi_1 = -2.0, phi_2 = 2.0, phi_3 = 1.5;

	torch::Tensor x1 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);
	torch::Tensor x2 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);
	std::vector<torch::Tensor> t = {x1, x2};
	auto mes = torch::meshgrid(t);  // https://www.geeksforgeeks.org/numpy-meshgrid-function/
	x1 = mes[1]; x2 = mes[0];
	std::cout << x1.index({Slice(0,10), Slice(0,10)}) << '\n';
	std::cout << x2.index({Slice(0,10), Slice(0,10)}) << '\n';

	// We run the neural network for each of these input values
	torch::Tensor y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3;
	std::tie(y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3) =
    shallow_2_1_3(x1,x2, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_12,
    				theta_20, theta_21, theta_22, theta_30, theta_31, theta_32);

	std::cout << y.sizes() << " x1: " << x1.sizes() << " pre_1: " << pre_1.sizes() << " act_1: " << act_1.sizes() << '\n';

	plot_neural_2_inputs(x1, x2, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3);

	std::cout << "// ---------------------------------------------------------------------------------\n";
	std::cout << "// Shallow neural network with, two inputs, two outputs, and three hidden units\n";
	std::cout << "// ---------------------------------------------------------------------------------\n";

	// Now lets define some parameters and run the neural network
	theta_10 =  -4.0,  theta_11 = 0.9, theta_12 = 0.0,
		   theta_20 =  5.0, theta_21 = -0.9, theta_22 = -0.5,
		   theta_30 =  -7.0, theta_31 = 0.5, theta_32 = 0.9;
	double phi_10 = 0.0, phi_11 = -2.0, phi_12 = 2.0, phi_13 = 1.5,
		   phi_20 = -2.0, phi_21 = -1.0, phi_22 = -2.0, phi_23 = 0.8;

	x1 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);
	x2 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);

	mes = torch::meshgrid({x1, x2});
	x1 = mes[1]; x2 = mes[0];

	// We run the neural network for each of these input values
	torch::Tensor  y1, y2, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23;

	std::tie(y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11, w_act_12, w_act_13, w_act_21, w_act_22, w_act_23)
		= shallow_2_2_3(x1,x2, ReLU, phi_10,phi_11,phi_12,phi_13, phi_20,phi_21,phi_22,phi_23,
						theta_10, theta_11, theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32);

	// And then plot it
	plot_neural_2_inputs_2_outputs(x1,x2, y1, y2, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_11,
			w_act_12, w_act_13, w_act_21, w_act_22, w_act_23);

	std::cout << "Done!\n";
	return 0;
}




