/*
 * UDL_util.cpp
 *
 *  Created on: Jan 9, 2025
 *      Author: jiamny
 */

#include "UDL_util.h"
#include "TempHelpFunctions.h"
#include "helpfunction.h"

#include <matplot/matplot.h>
using namespace matplot;

// Gabor model definition
torch::Tensor model(torch::Tensor phi, torch::Tensor x) {
	torch::Tensor sin_component = torch::sin(phi[0] + 0.06 * phi[1] * x);
	torch::Tensor gauss_component = torch::exp(-(phi[0] + 0.06 * phi[1] * x) * (phi[0] + 0.06 * phi[1] * x) / 32);
	torch::Tensor y_pred= sin_component * gauss_component;
  return y_pred;
}

// Draw model
void draw_model(torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor),
		torch::Tensor phi, std::string title) {
	torch::Tensor x_model = torch::arange(-15, 15, 0.1);
	torch::Tensor y_model = model(phi, x_model);

	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	matplot::axes_handle fx = F->nexttile();

	matplot::hold(fx, true);
	matplot::plot(fx, tensorTovector(data[0].to(torch::kDouble)),
					  tensorTovector(data[1].to(torch::kDouble)), "bo")->line_width(2);
	matplot::plot(fx, tensorTovector(x_model.to(torch::kDouble)),
					  tensorTovector(y_model.to(torch::kDouble)),"m-")->line_width(3);
	matplot::xlim({-15,15});
	matplot::ylim({-1, 1});
	matplot::xlabel("x");
	matplot::ylabel("y");

  if( title != "" )
	    matplot::title(fx, title);
  matplot::show();
}

torch::Tensor compute_loss(torch::Tensor data_x, torch::Tensor data_y,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi) {

	torch::Tensor pred_y = model(phi, data_x);
	torch::Tensor loss = torch::sum(torch::pow((pred_y - data_y), 2));

	return loss;
}

// These came from writing out the expression for the sum of squares loss and taking the
// derivative with respect to phi0 and phi1. It was a lot of hassle to get it right!
torch::Tensor gabor_deriv_phi0(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi0, torch::Tensor phi1) {
	torch::Tensor x = 0.06 * phi1 * data_x + phi0;
	torch::Tensor y = data_y.clone();

	torch::Tensor cos_component = torch::cos(x);
	torch::Tensor sin_component = torch::sin(x);
	torch::Tensor gauss_component = torch::exp(-0.5 * x *x / 16);
	torch::Tensor deriv = cos_component * gauss_component - sin_component * gauss_component * x / 16;
	deriv = 2* deriv * (sin_component * gauss_component - y);
    return torch::sum(deriv);
}

torch::Tensor gabor_deriv_phi1(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi0, torch::Tensor phi1) {
	torch::Tensor x = 0.06 * phi1 * data_x + phi0;
	torch::Tensor y = data_y.clone();

	torch::Tensor cos_component = torch::cos(x);
	torch::Tensor sin_component = torch::sin(x);
	torch::Tensor gauss_component = torch::exp(-0.5 * x *x / 16);
	torch::Tensor deriv = 0.06 * data_x * cos_component * gauss_component - 0.06 * data_x*sin_component * gauss_component * x / 16;
	deriv = 2*deriv * (sin_component * gauss_component - y);
    return torch::sum(deriv);
}

torch::Tensor compute_gradient(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi) {
	float dl_dphi0 = gabor_deriv_phi0(data_x, data_y, phi[0], phi[1]).data().item<float>();
	float dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0],phi[1]).data().item<float>();
    // Return the gradient
    return torch::tensor({{dl_dphi0}, {dl_dphi1}});
}

torch::Tensor gradient_descent_step(torch::Tensor phi, torch::Tensor data, float learning_rate,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor)) {
	// Step 1:  Compute the gradient
	torch::Tensor gradient = compute_gradient(data[0], data[1], phi);
	// Step 2:  Update the parameters
	torch::Tensor new_phi = phi - learning_rate * gradient;
	return new_phi;
}

void draw_loss_function(torch::Tensor (*compute_loss)(torch::Tensor, torch::Tensor,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor),
		torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi_iters){

	// Make grid of intercept/slope values to plot
	torch::Tensor intercepts_mesh, slopes_mesh;
    std::vector<torch::Tensor> grids = torch::meshgrid({torch::arange(-10,10.0,0.1), torch::arange(2.5, 22.5, 0.1)}, "ij");
    intercepts_mesh = grids[0];
    slopes_mesh = grids[1];
    torch::Tensor loss_mesh = torch::zeros_like(slopes_mesh);

    // Compute loss for every set of parameters
    int R = slopes_mesh.size(0), C = slopes_mesh.size(1);
    for(auto& r : range(R, 0)) {
    	for(auto& c : range(C, 0)) {
    		float intercept = intercepts_mesh[r][c].data().item<float>();
    		float slope = slopes_mesh[r][c].data().item<float>();
    		loss_mesh[r][c] = compute_loss(data[0], data[1], model, torch::tensor({{intercept}, {slope}})).data().item<float>();
    	}
    }

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < intercepts_mesh.size(0);  i += 1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (int j = 0; j < intercepts_mesh.size(1); j += 1) {
	            x_row.push_back(intercepts_mesh[i][j].data().item<double>());
	            y_row.push_back(slopes_mesh[i][j].data().item<double>());
	            z_row.push_back(loss_mesh[i][j].data().item<double>());
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

	matplot::axes_handle fx = F->nexttile();
	matplot::hold(fx, true);

	//std::vector<double> lvls = linspace(-10.0, 10.0, 10);
	matplot::contour(fx, X, Y, Z)->line_width(2); //.levels(lvls);

    if( phi_iters.numel() > 0 ) {
	    matplot::plot(fx, tensorTovector(phi_iters[0].to(torch::kDouble)),
			  	  	 tensorTovector(phi_iters[1].to(torch::kDouble)), "mo:")->line_width(3);
    }
	matplot::ylim(fx, {2.5, 22.5});
	matplot::xlabel(fx, "Offset phi-0");
	matplot::ylabel(fx, "Frequency, phi-1");
	matplot::show();
}


