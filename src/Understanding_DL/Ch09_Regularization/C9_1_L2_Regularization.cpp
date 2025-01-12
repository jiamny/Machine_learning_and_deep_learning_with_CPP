/*
 * C9_1_L2_Regularization.cpp
 *
 *  Created on: Dec 26, 2024
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

// if we had some weak knowledge that the solution was in the vicinity of 𝜙0=0.0, 𝜙1=12.5 (the center of the plot)?
// add a term to the loss function that penalizes solutions that deviate from this point.
// 𝐿′[𝝓] = 𝐿[𝝓] + 𝜆 * (𝜙0^2+(𝜙1−12.5)^2)
// Computes the regularization term
float compute_reg_term(float phi0, float phi1) {
    // compute the regularization term (term in large brackets in the above equation)
    float reg_term = std::pow(phi0, 2) + std::pow(phi1 - 12.5, 2);

    return reg_term;
}

//# Define the loss function
torch::Tensor compute_loss2(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor (*model)(torch::Tensor, torch::Tensor),
		torch::Tensor phi, float lambda_) {
	torch::Tensor pred_y = model(phi, data_x);
	torch::Tensor loss = torch::sum((pred_y-data_y)*(pred_y-data_y));
	// Add the new term to the loss
    loss = loss + lambda_ * compute_reg_term(phi[0].data().item<float>(), phi[1].data().item<float>());

  return loss;
}


// Code to draw the regularization function
void draw_reg_function() {

    // Make grid of offset/frequency values to plot
	std::vector<torch::Tensor> grids = torch::meshgrid({torch::arange(-10,10.0,0.1), torch::arange(2.5,22.5,0.1)}, "ij");
    torch::Tensor offsets_mesh = grids[0], freqs_mesh = grids[1];
    torch::Tensor loss_mesh = torch::zeros_like(freqs_mesh);
    // Compute loss for every set of parameters
	int R = freqs_mesh.size(0), C = freqs_mesh.size(1);
	for(auto& r : range(R, 0)) {
		for(auto& c : range(C, 0)) {
		    float intercept = offsets_mesh[r][c].data().item<float>();
		    float slope = freqs_mesh[r][c].data().item<float>();
		    loss_mesh[r][c] = compute_reg_term(intercept, slope);
		}
	}

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < offsets_mesh.size(0);  i += 1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (int j = 0; j < offsets_mesh.size(1); j += 1) {
	            x_row.push_back(offsets_mesh[i][j].data().item<double>());
	            y_row.push_back(freqs_mesh[i][j].data().item<double>());
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

	//matplot::contour(fx, X, Y, Z)->line_width(2).levels(lvls);
	matplot::contour(fx, X, Y, Z)->line_width(2);
	matplot::ylim(fx, {2.5, 22.5});
	matplot::xlabel(fx, "Offset phi-0");
	matplot::ylabel(fx, "Frequency, phi-1");
	matplot::show();

}

// Code to draw loss function with regularization
void draw_loss_function_reg(torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor),
		float lambda_, torch::Tensor phi_iters = torch::empty(0)) {

    // Make grid of offset/frequency values to plot
	std::vector<torch::Tensor> grids = torch::meshgrid({torch::arange(-10,10.0,0.1), torch::arange(2.5,22.5,0.1)}, "ij");
    torch::Tensor offsets_mesh = grids[0], freqs_mesh = grids[1];
    torch::Tensor loss_mesh = torch::zeros_like(freqs_mesh);
    torch::Tensor phi = torch::zeros({2, 1});
    // Compute loss for every set of parameters
	int R = freqs_mesh.size(0), C = freqs_mesh.size(1);
	for(auto& r : range(R, 0)) {
		for(auto& c : range(C, 0)) {
			phi[0] = offsets_mesh[r][c].data().item<float>();
			phi[1] = freqs_mesh[r][c].data().item<float>();
		    loss_mesh[r][c] = compute_loss2(data.index({0, Slice()}), data.index({1, Slice()}), model, phi, lambda_);
		}
	}

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < offsets_mesh.size(0);  i += 1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (int j = 0; j < offsets_mesh.size(1); j += 1) {
	            x_row.push_back(offsets_mesh[i][j].data().item<double>());
	            y_row.push_back(freqs_mesh[i][j].data().item<double>());
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

	matplot::contour(fx, X, Y, Z)->line_width(2);
    if( phi_iters.numel() > 0 ) {
	    matplot::plot(tensorTovector(phi_iters[0].to(torch::kDouble)),
			  	  	 tensorTovector(phi_iters[1].to(torch::kDouble)), "mo:")->line_width(3);
    }
	matplot::ylim(fx, {2.5, 22.5});
	matplot::xlabel(fx, "Offset phi-0");
	matplot::ylabel(fx, "Frequency, phi-1");
	matplot::show();
}

// 𝜆 * (𝜙0^2+(𝜙1−12.5)^2)
torch::Tensor dldphi0(torch::Tensor phi, float lambda_){
  // compute the derivative with respect to phi0
	torch::Tensor deriv = 2*lambda_*phi[0];

    return deriv;
}

torch::Tensor dldphi1(torch::Tensor phi, float lambda_) {
    // compute the derivative with respect to phi1
	torch::Tensor deriv = lambda_ *(2*phi[1] - 2*12.5);

    return deriv;
}

torch::Tensor compute_gradient2(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor  phi, float lambda_) {
	torch::Tensor dl_dphi0 = gabor_deriv_phi0(data_x, data_y, phi[0], phi[1]) + dldphi0(torch::squeeze(phi), lambda_);
	torch::Tensor dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0], phi[1]) + dldphi1(torch::squeeze(phi), lambda_);
    // Return the gradient
    return torch::tensor({{dl_dphi0.data().item<float>()}, {dl_dphi1.data().item<float>()}});
}

torch::Tensor gradient_descent_step2(torch::Tensor phi, float lambda_, torch::Tensor data,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor)) {
    // Step 1:  Compute the gradient
	torch::Tensor gradient = compute_gradient2(data.index({0, Slice()}), data.index({1, Slice()}), phi, lambda_);
    // Step 2:  Update the parameters -- note we want to search in the negative (downhill direction)
    float alpha = 0.1;
    phi = phi - alpha * gradient;
    return phi;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Fit the Gabor model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
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

	// Initialize the parameters and draw the model
	torch::Tensor phi = torch::zeros({2, 1});
	phi[0] =  -5;     							// Horizontal offset
	phi[1] =  25;     							// Frequency
	if(plt) draw_model(data, model, phi, "Initial parameters");

	torch::Tensor loss = compute_loss(data[0], data[1], model, torch::tensor({{0.6}, {-0.2}}));
	printf("Your loss = %3.3f, Correct loss = %3.3f\n", loss.data().item<float>(), 16.419);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw loss function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	if(plt) draw_loss_function(compute_loss, data, model);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "//  Measure loss and draw initial model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Initialize the parameters
	int n_steps = 40;
	torch::Tensor phi_all = torch::zeros({2,n_steps+1});
	phi_all[0][0] = 2.6;
	phi_all[1][0] = 8.5;
	float learning_rate = 0.1;

	// Measure loss and draw initial model
	loss =  compute_loss(data.index({0, Slice()}), data.index({0, Slice()}), model, phi_all.index({Slice(), Slice(0, 1)}));
	if(plt) {
		draw_model(data, model, phi_all.index({Slice(), Slice(0, 1)}),
			"Initial parameters, Loss = " + std::to_string(loss.data().item<float>()));
	}

	for(auto& c_step : range (n_steps, 0)) {
	    // Do gradient descent step
	    phi_all.index_put_( {Slice(), Slice(c_step+1, c_step+2)},
			  gradient_descent_step(phi_all.index({Slice(), Slice(c_step, c_step+1)}), data, learning_rate, model));

	    // Measure loss and draw model every 8th step
	    if( (c_step+1) % 8 == 0 ) {
	    	loss =  compute_loss(data.index({0, Slice()}), data.index({1, Slice()}),
	    		model, phi_all.index({Slice(), Slice(c_step+1, c_step+2)}));

	    	if(plt) {
	    		draw_model(data,model,phi_all.index({Slice(), c_step+1}),
	    		"Iteration " + std::to_string(c_step+1) + ", loss = " + std::to_string(loss.data().item<float>()));
	    	}
	    }
	}

	if(plt) draw_loss_function(compute_loss, data, model, phi_all);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw the regularization function.  It should look similar to figure 9.1b\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	draw_reg_function();

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw the regularization function.  It looks something like figure 9.1c\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	float lambda_ = 0.2;
	draw_loss_function_reg(data,  model, lambda_);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Finally, let's run gradient descent and draw the result\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Initialize the parameters
	n_steps = 40;
	phi_all[0][0] = 2.6;
	phi_all[1][0] = 8.5;
	lambda_ = 0.2;

	// Measure loss and draw initial model
	loss =  compute_loss2(data.index({0, Slice()}), data.index({1, Slice()}),
    						model, phi_all.index({Slice(), Slice(0, 1)}), lambda_);

	draw_model(data, model, phi_all.index({Slice(), Slice(0,1)}),
			"Initial parameters, Loss = " + std::to_string(loss.data().item<float>()));

	for(auto& c_step : range (n_steps, 0)) {
	  // Do gradient descent step
	  //phi_all[:,c_step+1:c_step+2] = gradient_descent_step2(phi_all[:,c_step:c_step+1],lambda_, data, model)
	    phi_all.index_put_( {Slice(), Slice(c_step+1, c_step+2)},
				  gradient_descent_step2(phi_all.index({Slice(), Slice(c_step, c_step+1)}), lambda_, data, model));

	    // Measure loss and draw model every 8th step
	    if( (c_step+1) % 8 == 0 ) {
	    	loss =  compute_loss2(data.index({0, Slice()}), data.index({1, Slice()}),
	    		model, phi_all.index({Slice(), Slice(c_step+1, c_step+2)}), lambda_);

	    	draw_model(data, model, phi_all.index({Slice(), c_step+1}),
	    		"Iteration " + std::to_string(c_step+1) + ", loss = " + std::to_string(loss.data().item<float>()));
	    }
	}

	draw_loss_function_reg(data, model, lambda_, phi_all);

	std::cout << "Done!\n";
	return 0;
}




