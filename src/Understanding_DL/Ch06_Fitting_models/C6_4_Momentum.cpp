/*
 * C6_4_Momentum.cpp
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

// define model
torch::Tensor model(torch::Tensor phi, torch::Tensor x) {
	torch::Tensor sin_component = torch::sin(phi[0] + 0.06 * phi[1] * x);
	torch::Tensor gauss_component = torch::exp(-(phi[0] + 0.06 * phi[1] * x) * (phi[0] + 0.06 * phi[1] * x) / 32);
	torch::Tensor y_pred= sin_component * gauss_component;
	return y_pred;
}

// Draw model
void draw_model(torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi, std::string title="") {
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
	matplot::xlim(fx, {-15,15});
	matplot::ylim(fx, {-1, 1});
	matplot::xlabel(fx, "x");
	matplot::ylabel(fx, "y");

  if( title != "" )
	    matplot::title(fx, title);
  matplot::show();
}

torch::Tensor compute_loss(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi) {
	torch::Tensor pred_y = model(phi, data_x);
	torch::Tensor loss = torch::sum((pred_y-data_y)*(pred_y-data_y));
	return loss;
}

void draw_loss_function(torch::Tensor (*compute_loss)(torch::Tensor, torch::Tensor,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor), torch::Tensor data,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi_iters = torch::empty(0), std::string title="") {
	// Make grid of intercept/slope values to plot
	torch::Tensor intercepts_mesh, slopes_mesh;
    std::vector<torch::Tensor> grids = torch::meshgrid({torch::arange(2.5, 22.5, 0.1), torch::arange(-10,10.0,0.1)}, "ij");
    slopes_mesh = grids[0];
    intercepts_mesh = grids[1];
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
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	matplot::axes_handle fx = F->nexttile();

	matplot::hold(fx, true);
	matplot::contour(fx, X, Y, Z)->line_width(2);

	if( phi_iters.numel() > 0) {
		matplot::plot(fx, tensorTovector(phi_iters[0].to(torch::kDouble)),
		  	  	 tensorTovector(phi_iters[1].to(torch::kDouble)), "mo:")->line_width(3);
	}
	matplot::ylim(fx, {2.5, 22.5});
	matplot::xlabel(fx, "Offset phi-0");
	matplot::ylabel(fx, "Frequency, phi-1");
	if( title != "" )
		matplot::title(fx, title);
	matplot::show();

}

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
    float dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0], phi[1]).data().item<float>();
    // Return the gradient
    return torch::tensor({{dl_dphi0}, {dl_dphi1}});
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = true;

	// We'll try to fit the Gabor model to these data
	torch::Tensor data = torch::tensor({{-1.920e+00,-1.422e+01,1.490e+00,-1.940e+00,-2.389e+00,-5.090e+00,
                 -8.861e+00,3.578e+00,-6.010e+00,-6.995e+00,3.634e+00,8.743e-01,
                 -1.096e+01,4.073e-01,-9.467e+00,8.560e+00,1.062e+01,-1.729e-01,
                  1.040e+01,-1.261e+01,1.574e-01,-1.304e+01,-2.156e+00,-1.210e+01,
                 -1.119e+01,2.902e+00,-8.220e+00,-1.179e+01,-8.391e+00,-4.505e+00},
                  {-1.051e+00,-2.482e-02,8.896e-01,-4.943e-01,-9.371e-01,4.306e-01,
                  9.577e-03,-7.944e-02 ,1.624e-01,-2.682e-01,-3.129e-01,8.303e-01,
                  -2.365e-02,5.098e-01,-2.777e-01,3.367e-01,1.927e-01,-2.222e-01,
                  6.352e-02,6.888e-03,3.224e-02,1.091e-02,-5.706e-01,-5.258e-02,
                  -3.666e-02,1.709e-01,-4.805e-02,2.008e-01,-1.904e-01,5.952e-01}});

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw this model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Initialize the parameters to some arbitrary values and draw the model
	torch::Tensor phi = torch::zeros({2,1});
	phi[0][0] = -5;     // Horizontal offset
	phi[1][0] = 25;     // Frequency
	if(plt) draw_model(data, model, phi, "Initial parameters");

	draw_loss_function(compute_loss, data, model);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Measure loss and draw model with fixed alpha\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	int n_steps = 81;
	int batch_size = 5;
	float alpha = 0.6;
	torch::Tensor phi_all = torch::zeros({2,n_steps+1});
	phi_all[0][0] = -1.5;
	phi_all[1][0] = 6.5;

	for(auto& c_step : range (n_steps, 0)) {
	    // Choose random batch indices
		torch::Tensor batch_index = torch::randint(0, data.size(1), {batch_size}); //np.random.permutation(data.shape[1])[0:batch_size]
	    // Compute the gradient
		torch::Tensor gradient = compute_gradient(data.index({0,batch_index}),
				data.index({1,batch_index}), phi_all.index({Slice(), Slice(c_step, c_step+1)}));
	    // Update the parameters
		torch::Tensor new_phi = phi_all.index({Slice(), Slice(c_step, c_step+1)}) - alpha * gradient;

		phi_all.index_put_({Slice(), Slice(c_step+1, c_step+2)}, new_phi);
	}

	torch::Tensor loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), n_steps}));
	std::string tlt = "Iteration " + std::to_string(n_steps) + ", loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), n_steps}), tlt);
	tlt = "SGD with fixed learning rate";
	draw_loss_function(compute_loss, data, model, phi_all, tlt);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Measure loss and draw model with momentum\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	n_steps = 81;
	batch_size = 5;
	alpha = 0.6;
	float beta = 0.6;
	torch::Tensor momentum = torch::zeros({2, 1});
	phi_all = torch::zeros({2, n_steps+1});
	phi_all[0][0] = -1.5;
	phi_all[1][0] = 6.5;

	// Measure loss and draw initial model
	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), 0}));
	tlt = "Iteration " + std::to_string(n_steps) + ", loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), 0}), tlt);

	for(auto& c_step : range (n_steps, 0)) {
	    // Choose random batch indices
		torch::Tensor batch_index = torch::randint(0, data.size(1), {batch_size}); //np.random.permutation(data.shape[1])[0:batch_size]
	    // Compute the gradient
		torch::Tensor gradient = compute_gradient(data.index({0,batch_index}),
				data.index({1,batch_index}), phi_all.index({Slice(), Slice(c_step, c_step+1)}));
		//---------------------------------------------------------------------------------------
		// Update the momentum
		//---------------------------------------------------------------------------------------
		momentum = beta * momentum + (1.0 - beta) * gradient;
	    // Update the parameters
		torch::Tensor new_phi = phi_all.index({Slice(), Slice(c_step, c_step+1)}) - alpha * momentum;

		phi_all.index_put_({Slice(), Slice(c_step+1, c_step+2)}, new_phi);
	}

	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), n_steps}));
	tlt = "Iteration " + std::to_string(n_steps) + ", loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), n_steps}), tlt);
	tlt = "SGD with momentum";
	draw_loss_function(compute_loss, data, model, phi_all, tlt);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Measure loss and draw model with Nesterov momentum\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	n_steps = 81;
	batch_size = 5;
	alpha = 0.6;
	beta = 0.6;
	momentum = torch::zeros({2, 1});
	phi_all = torch::zeros({2, n_steps+1});
	phi_all[0][0] = -1.5;
	phi_all[1][0] = 6.5;

	// Measure loss and draw initial model
	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), 0}));
	tlt = "Iteration " + std::to_string(n_steps) + ", loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), 0}), tlt);

	for(auto& c_step : range (n_steps, 0)) {
	    // Choose random batch indices
		torch::Tensor batch_index = torch::randint(0, data.size(1), {batch_size}); //np.random.permutation(data.shape[1])[0:batch_size]

		// Compute the gradient
		torch::Tensor _phi = phi_all.index({Slice(), Slice(c_step, c_step+1)}) - alpha * beta * momentum;

		torch::Tensor gradient = compute_gradient(data.index({0, batch_index}),
				data.index({1, batch_index}), _phi);
		//---------------------------------------------------------------------------------------
		// Update the momentum
		//---------------------------------------------------------------------------------------
		momentum = beta * momentum + (1.0 - beta) * gradient;
	    // Update the parameters
		torch::Tensor new_phi = phi_all.index({Slice(), Slice(c_step, c_step+1)}) - alpha * momentum;

		phi_all.index_put_({Slice(), Slice(c_step+1, c_step+2)}, new_phi);
	}

	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), n_steps}));
	tlt = "Iteration " + std::to_string(n_steps) + ", loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), n_steps}), tlt);
	tlt = "SGD with Nesterov momentum";
	draw_loss_function(compute_loss, data, model, phi_all, tlt);

	std::cout << "Done!\n";
	return 0;
}


