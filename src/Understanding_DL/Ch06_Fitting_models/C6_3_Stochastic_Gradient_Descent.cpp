/*
 * C6_3_Stochastic_Gradient_Descent.cpp
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
#include "../../Utils/UDL_util.h"
#include "../../Utils/TempHelpFunctions.h"

#include <matplot/matplot.h>
using namespace matplot;



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

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

	torch::Tensor loss = compute_loss(data[0], data[1], model, torch::tensor({{0.6}, {-0.2}}));
	printf("Your loss = %3.3f, Correct loss = %3.3f\n", loss.data().item<float>(), 16.419);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw loss function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	draw_loss_function(compute_loss, data, model);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute the gradient\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor gradient = compute_gradient(data[0], data[1], phi);
	printf("Your gradients: (%3.3f, %3.3f)\n", gradient[0].data().item<float>(), gradient[1].data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Approximate the gradients with finite differences\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	//Approximate the gradients with finite differences
	float delta = 0.0001;
	torch::Tensor dl_dphi0_est = (compute_loss(data[0], data[1], model,
			phi + torch::tensor({{delta}, {0.0f}}).to(torch::kFloat32)) -
	        compute_loss(data[0], data[1], model, phi)) / delta;
	torch::Tensor dl_dphi1_est = (compute_loss(data[0], data[1], model,
			phi + torch::tensor({{0.0f}, {delta}}).to(torch::kFloat32)) -
	        compute_loss(data[0], data[1],model,phi))/delta;

	printf("Approx gradients: (%3.3f, %3.3f)\n", dl_dphi0_est.data().item<float>(), dl_dphi1_est.data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Measure loss and draw initial model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Initialize the parameters
	int n_steps = 1000;
	int save_phi_per_steps = 100;
	torch::Tensor phi_all = torch::zeros({2, n_steps+1});
	phi_all[0][0] = -1.6;
	phi_all[1][0] = 8.5;
	float learning_rate = 0.01;

	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), Slice(0, 1)}));
	std::string tlt = "Initial parameters, Loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), Slice(0, 1)}), tlt);

	int slt_phi_n = static_cast<int>(n_steps *1.0 / save_phi_per_steps);
	torch::Tensor slt_phi_all = torch::zeros({2, slt_phi_n});

	int j = 0;
	// Repeatedly take gradient descent steps
	for(auto& c_step : range (n_steps, 0)) {
	    // Do gradient descent step
	    torch::Tensor t = gradient_descent_step(phi_all.index({Slice(), Slice(c_step, c_step+1)}), data, learning_rate, model);
	    phi_all.index_put_({Slice(), Slice(c_step+1, c_step+2)}, t);
	    // Measure loss and draw model
	    loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), Slice(c_step+1, c_step+2)}));

	    if( c_step % save_phi_per_steps == 0) {
	    	std::string tlt = "Iteration " + std::to_string(c_step+1) + ", loss = " + std::to_string(loss.data().item<float>());
	    	draw_model(data, model, phi_all.index({Slice(), c_step+1}), tlt);
	    	slt_phi_all.index_put_({Slice(), Slice(j, j+1)}, t);
	    	j++;
	    }
	}

	// Draw the trajectory on the loss function
	std::cout << slt_phi_all << '\n';
	draw_loss_function(compute_loss, data, model, slt_phi_all);

	std::cout << "Done!\n";
	return 0;
}


