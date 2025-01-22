/*
 * C9_2_Implicit_regularization.cpp
 *
 *  Created on: Jan 9, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/UDL_util.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// define loss function
float loss(float phi0, float phi1) {
    float phi1_std = std::exp(-0.5 * (phi0 * phi0)*4.0);
    return 1.0 - std::exp(-0.5 * (phi1 * phi1)/(phi1_std * phi1_std));
}

// Compute the gradient (just done with finite differences for simplicity)
torch::Tensor get_loss_gradient(float phi0, float phi1) {
    float delta_phi = 0.00001;
    torch::Tensor gradient = torch::zeros({2,1});
    gradient[0][0] = (loss(phi0 + delta_phi/2.0, phi1) - loss(phi0 - delta_phi/2.0, phi1))/delta_phi;
    gradient[1][0] = (loss(phi0, phi1+delta_phi/2.0) - loss(phi0, phi1 - delta_phi/2.0))/delta_phi;
    return gradient;
}

// Perform gradient descent n_steps times and return path
torch::Tensor grad_descent(torch::Tensor start_posn, int n_steps, float step_size) {
	torch::Tensor grad_path = torch::zeros({2, n_steps+1});
    grad_path.index_put_({Slice(), 0}, start_posn.index({Slice(), 0})); //[:,0] = start_posn[:,0];

    for(auto& c_step : range(n_steps, 0) ) {
    	torch::Tensor this_grad = get_loss_gradient(grad_path[0][c_step].data().item<float>(),
    												grad_path[1][c_step].data().item<float>());

    	grad_path.index_put_({Slice(), c_step+1},
    			grad_path.index({Slice(), c_step}) - step_size * this_grad.index({Slice(), 0}));
    }
    return grad_path;
}

// Draw the loss function and the trajectories on it
void draw_function(torch::Tensor phi0mesh, torch::Tensor phi1mesh, torch::Tensor loss_function,
		torch::Tensor grad_path_tiny_lr=torch::empty(0), torch::Tensor grad_path_typical_lr=torch::empty(0)) {

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < phi0mesh.size(0);  i += 1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (int j = 0; j < phi0mesh.size(1); j += 1) {
	            x_row.push_back(phi0mesh[i][j].data().item<double>());
	            y_row.push_back(phi1mesh[i][j].data().item<double>());
	            z_row.push_back(loss_function[i][j].data().item<double>());
	    }
	    X.push_back(x_row);
	    Y.push_back(y_row);
	    Z.push_back(z_row);
	}

	auto F = figure(true);
	F->size(800, 800);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);

	matplot::contour(fx, X, Y, Z)->line_width(2);

	if( grad_path_typical_lr.numel() > 0 ) {
		matplot::plot(fx, tensorTovector(grad_path_typical_lr[0].to(torch::kDouble)),
				tensorTovector(grad_path_typical_lr[1].to(torch::kDouble)),"ro-")->line_width(2);
	}

	if( grad_path_tiny_lr.numel() > 0 ) {
		matplot::plot(fx, tensorTovector(grad_path_tiny_lr[0].to(torch::kDouble)),
				tensorTovector(grad_path_tiny_lr[1].to(torch::kDouble)), "b--")->line_width(3);
	}

	matplot::xlabel(fx, "phi-0");
	matplot::ylabel(fx, "phi-1");
	matplot::show();
}

// Compute the implicit regularization term (second term in equation 9.8 in the book)
float get_reg_term(float phi0, float phi1, float alpha){
    // compute this term
    // You can use the function get_loss_gradient(phi0, phi1) that was defined above
	torch::Tensor ls = get_loss_gradient(phi0, phi1);

	float ls0 = ls[0][0].data().item<float>();
	float ls1 = ls[1][0].data().item<float>();
	float reg_term = (alpha/4.0)*std::sqrt(ls0*ls0 + ls1*ls1);

    return reg_term;
}

// Compute modified loss function (equation 9.8)
float loss_reg(float phi0, float phi1, float alpha) {
    // The original function
	float phi1_std = std::exp(-0.5 * (phi0 * phi0)*4.0);
    float loss_out =  1.0 - std::exp(-0.5 * (phi1 * phi1)/(phi1_std * phi1_std));
    // Add the regularization term that you just calculated above
    loss_out = loss_out + get_reg_term(phi0, phi1, alpha);
    return loss_out;
}

// Compute gradient of modified loss function for gradient descent
torch::Tensor get_loss_gradient_reg(float phi0, float phi1, float alpha) {
    float delta_phi = 0.0001;
    torch::Tensor gradient = torch::zeros({2, 1});

    gradient[0][0] = (loss_reg(phi0 + delta_phi/2.0, phi1, alpha) - loss_reg(phi0 - delta_phi/2.0, phi1, alpha))/delta_phi;
    gradient[1][0] = (loss_reg(phi0, phi1 + delta_phi/2.0, alpha) - loss_reg(phi0, phi1 - delta_phi/2.0, alpha))/delta_phi;
    return gradient;
}

// Perform gradient descent n_steps times on modified loss function and return path
// Alpha is the step size for the gradient descent
// Alpha reg is the step size used to calculate the regularization term
torch::Tensor grad_descent_reg(torch::Tensor start_posn, int n_steps, float alpha, float alpha_reg) {
	torch::Tensor grad_path = torch::zeros({2, n_steps+1});
	grad_path.index_put_({Slice(), 0}, start_posn.index({Slice(), 0}));

    for(auto& c_step : range(n_steps, 0)) {
    	torch::Tensor this_grad = get_loss_gradient_reg(grad_path[0][c_step].data().item<float>(),
    							grad_path[1][c_step].data().item<float>(), alpha_reg);

        grad_path.index_put_({Slice(), c_step+1},
        		grad_path.index({Slice(), c_step}) - alpha * this_grad.index({Slice(),0}));
    }
    return grad_path;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU available." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw the loss function and the trajectories on it\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// define grid to plot function
	torch::Tensor grid_values = torch::arange(-0.8,0.5,0.01);

	auto rlt = torch::meshgrid({grid_values, grid_values}, "ij");
	torch::Tensor phi0mesh = rlt[1], phi1mesh = rlt[0];

	torch::Tensor loss_function = torch::zeros({grid_values.size(0), grid_values.size(0)});
	for(auto& idphi0 : range(static_cast<int>(grid_values.size(0)), 0)) {
		float phi0 = grid_values[idphi0].data().item<float>();
    	//for idphi1, phi1 in enumerate(grid_values):
		for(auto& idphi1 : range(static_cast<int>(grid_values.size(0)), 0)) {
			float phi1 = grid_values[idphi1].data().item<float>();
			loss_function[idphi0][idphi1] = loss(phi1,phi0);
		}
	}

	// Define the start position
	torch::Tensor start_posn = torch::zeros({2,1});
	start_posn[0][0] = -0.7; start_posn[1][0] = -0.75;

	// Run the gradient descent with a very small learning rate to simulate continuous case
	torch::Tensor grad_path_tiny_lr = grad_descent(start_posn, 10000, 0.001);

	// Run the gradient descent with a typical sized learning rate
	torch::Tensor grad_path_typical_lr = grad_descent(start_posn, 100, 0.05);

	if(plt) draw_function(phi0mesh, phi1mesh, loss_function, grad_path_tiny_lr, grad_path_typical_lr);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's visualize the regularization term\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	float alpha = 0.1;
	torch::Tensor reg_term = torch::zeros({grid_values.size(0), grid_values.size(0)});

	for(auto& idphi0 : range(static_cast<int>(grid_values.size(0)), 0)) {
		float phi0 = grid_values[idphi0].data().item<float>();
    	//for idphi1, phi1 in enumerate(grid_values):
		for(auto& idphi1 : range(static_cast<int>(grid_values.size(0)), 0)) {
			float phi1 = grid_values[idphi1].data().item<float>();
			reg_term[idphi0][idphi1] = get_reg_term(phi1,phi0, alpha);
		}
	}

	if(plt) draw_function(phi0mesh, phi1mesh, reg_term);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// We'll also visualize the loss function plus the regularization term\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// We'll also visualize the loss function plus the regularization term
	alpha = 0.1;
	torch::Tensor  loss_function_reg = torch::zeros({grid_values.size(0), grid_values.size(0)});

	for(auto& idphi0 : range(static_cast<int>(grid_values.size(0)), 0)) {
		float phi0 = grid_values[idphi0].data().item<float>();
    	//for idphi1, phi1 in enumerate(grid_values):
		for(auto& idphi1 : range(static_cast<int>(grid_values.size(0)), 0)) {
			float phi1 = grid_values[idphi1].data().item<float>();
			loss_function_reg[idphi0][idphi1] = loss_reg(phi1, phi0, alpha);
		}
	}

	if(plt) draw_function(phi0mesh, phi1mesh, loss_function_reg);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Perform gradient descent n_steps times on modified loss function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	start_posn[0][0] = -0.7;
	start_posn[1][0] = -0.75;

	//function with 10000 steps and alpha_reg = 0.05, and a very small learning rate alpha of 0.001
	grad_path_tiny_lr = grad_descent_reg(start_posn, 10000, 0.001, 0.05);

 	// function with 100 steps and a very small learning rate alpha of 0.05
	grad_path_typical_lr = grad_descent_reg(start_posn, 100, 0.05, 0.05);

	// Draw the functions
	draw_function(phi0mesh, phi1mesh, loss_function_reg, grad_path_tiny_lr, grad_path_typical_lr);

	std::cout << "Done!\n";
	return 0;
}




