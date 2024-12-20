/*
 * C6_5_Adam.cpp
 *
 *  Created on: Oct 20, 2024
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

// Define function that we wish to find the minimum of (normally would be defined implicitly by data and loss)
float loss(float phi0, float phi1) {
	float height = std::exp(-0.5 * (phi1 * phi1)*4.0);
    height = height * std::exp(-0.5* (phi0-0.7) *(phi0-0.7)/4.0);
    return 1.0 - height;
}

// Compute the gradients of this function (for simplicity, I just used finite differences)
torch::Tensor get_loss_gradient(float phi0, float phi1) {
    float delta_phi = 0.00001;
    torch::Tensor gradient = torch::zeros({2, 1});
    gradient[0] = (loss(phi0+delta_phi/2.0, phi1) - loss(phi0-delta_phi/2.0, phi1))/delta_phi;
    gradient[1] = (loss(phi0, phi1+delta_phi/2.0) - loss(phi0, phi1-delta_phi/2.0))/delta_phi;
    return gradient.index({Slice(), 0});
}

// Compute the loss function at a range of values of phi0 and phi1 for plotting
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_loss_function_for_plot() {
	torch::Tensor grid_values = torch::arange(-1.0, 1.0, 0.01);
	torch::Tensor phi0mesh, phi1mesh;
	std::vector<torch::Tensor> dt = torch::meshgrid({grid_values, grid_values}, "ij");
	phi0mesh = dt[1];
	phi1mesh = dt[0];

	torch::Tensor loss_function = torch::zeros({grid_values.size(0), grid_values.size(0)});
    for(auto& idphi0 : range(static_cast<int>(grid_values.size(0)), 0)) {
    	float phi0 = grid_values[idphi0].data().item<float>();
        for(auto& idphi1 : range(static_cast<int>(grid_values.size(0)), 0)) {
        	float phi1 = grid_values[idphi1].data().item<float>();
            loss_function[idphi0][idphi1] = loss(phi1, phi0);
        }
    }
  return std::make_tuple(loss_function, phi0mesh, phi1mesh);
}

//Plotting function
void draw_function(torch::Tensor phi0mesh, torch::Tensor phi1mesh, torch::Tensor loss_function, torch::Tensor opt_path) {

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < phi0mesh.size(0); i++) {
		std::vector<double> x_row, y_row, z_row;
		for(int j = 0; j < phi0mesh.size(1); j++) {
			x_row.push_back(phi0mesh[i][j].data().item<double>());
			y_row.push_back(phi1mesh[i][j].data().item<double>());
			z_row.push_back(loss_function[i][j].data().item<double>());
		}
		X.push_back(x_row);
		Y.push_back(y_row);
		Z.push_back(z_row);
	}

	auto F = figure(true);
	F->size(700, 700);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	matplot::axes_handle fx = F->nexttile();
	matplot::hold(fx, true);

	matplot::contour(fx, X, Y, Z)->line_width(2);
	matplot::plot(tensorTovector(opt_path[0].to(torch::kDouble)),
					tensorTovector(opt_path[1].to(torch::kDouble)), "m:o")->line_width(2).marker_size(6);

	matplot::xlabel("phi0");
	matplot::ylabel("phi1");
	matplot::show();
}

// Simple fixed step size gradient descent
torch::Tensor grad_descent(torch::Tensor start_posn, int n_steps, float alpha) {
	torch::Tensor grad_path = torch::zeros({2, n_steps+1});
    grad_path.index_put_({Slice(), 0}, start_posn.index({Slice(), 0}));

    for(auto& c_step : range(n_steps, 0)) {
    	torch::Tensor this_grad = get_loss_gradient(grad_path[0][c_step].data().item<float>(),
        											grad_path[1][c_step].data().item<float>());
        grad_path.index_put_({Slice(),c_step+1}, grad_path.index({Slice(), c_step}) - alpha * this_grad);
    }
    return grad_path;
}

torch::Tensor normalized_gradients(torch::Tensor start_posn, int n_steps, float alpha,  float epsilon=1e-20) {
	torch::Tensor grad_path = torch::zeros({2, n_steps+1});

	grad_path.index_put_({Slice(), 0}, start_posn.index({Slice(), 0}));

    for(auto& c_step : range(n_steps, 0)) {
        // Measure the gradient as in equation 6.13 (first line)
    	torch::Tensor m = get_loss_gradient(grad_path[0][c_step].data().item<float>(),
											grad_path[1][c_step].data().item<float>());

        // TO DO -- compute the squared gradient as in equation 6.13 (second line)
    	torch::Tensor v = torch::pow(m, 2);

        // TO DO -- apply the update rule (equation 6.14)
    	grad_path.index_put_({Slice(),c_step+1},
    			grad_path.index({Slice(), c_step}) - alpha * ( m / (torch::sqrt(v)+epsilon)));
    }

    return grad_path;
}

torch::Tensor adam(torch::Tensor start_posn, int n_steps, float alpha,  float beta = 0.9,
							float gamma = 0.99, float epsilon = 1e-20) {
	torch::Tensor grad_path = torch::zeros({2, n_steps+1});
	grad_path.index_put_({Slice(), 0}, start_posn.index({Slice(), 0}));

	torch::Tensor m = torch::zeros_like(grad_path.index({Slice(), 0}));
	torch::Tensor v = torch::zeros_like(grad_path.index({Slice(), 0}));

    for(auto& c_step : range(n_steps, 0)) {
        // Measure the gradient
    	torch::Tensor grad = get_loss_gradient(grad_path[0][c_step].data().item<float>(),
											   grad_path[1][c_step].data().item<float>());
        // TODO -- Update the momentum based gradient estimate equation 6.15 (first line)
		m = beta * m + (1.0 - beta) * grad;

        // TODO -- update the momentum based squared gradient estimate as in equation 6.15 (second line)
    	v = gamma * v + (1.0 - gamma) * torch::pow(grad, 2);

        // TODO -- Modify the statistics according to equation 6.16
    	torch::Tensor m_tilde = m / (1.0 - std::pow(beta, c_step + 1));
    	torch::Tensor v_tilde = v / (1.0 - std::pow(gamma, c_step + 1));

        // TO DO -- apply the update rule (equation 6.17)
    	grad_path.index_put_({Slice(),c_step+1},
    	    			grad_path.index({Slice(), c_step}) - alpha * (m_tilde / (torch::sqrt(v_tilde)+epsilon)));
    }

    return grad_path;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = true;
	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Run gradient descent\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor loss_function, phi0mesh, phi1mesh;
	std::tie(loss_function, phi0mesh, phi1mesh) = get_loss_function_for_plot();

	torch::Tensor start_posn = torch::zeros({2, 1});
	start_posn[0][0] = -0.7; start_posn[1][0] = -0.9;

	// Run gradient descent
	torch::Tensor grad_path1 = grad_descent(start_posn, 200, 0.08);
	draw_function(phi0mesh, phi1mesh, loss_function, grad_path1);
	torch::Tensor grad_path2 = grad_descent(start_posn, 40, 1.0);
	draw_function(phi0mesh, phi1mesh, loss_function, grad_path2);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's try out normalized gradients\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Let's try out normalized gradients
	start_posn = torch::zeros({2, 1});
	start_posn[0][0] = -0.7; start_posn[1][0] = -0.9;

	// Run gradient descent
	grad_path1 = normalized_gradients(start_posn, 40, 0.08);
	draw_function(phi0mesh, phi1mesh, loss_function, grad_path1);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's try out our Adam algorithm\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	start_posn = torch::zeros({2, 1});
	start_posn[0][0] = -0.7; start_posn[1][0] = -0.9;

	// Run gradient descent
	grad_path1 = adam(start_posn, 60, 0.05);
	draw_function(phi0mesh, phi1mesh, loss_function, grad_path1);

	std::cout << "Done!\n";
	return 0;
}


