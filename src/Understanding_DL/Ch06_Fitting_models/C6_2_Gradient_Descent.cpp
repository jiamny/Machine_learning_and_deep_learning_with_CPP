/*
 * C6_2_Gradient_Descent.cpp
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

// Let's define our model -- just a straight line with intercept phi[0] and slope phi[1]
torch::Tensor model(torch::Tensor phi, torch::Tensor x) {
	torch::Tensor y_pred = phi[0] + phi[1] * x;
    return y_pred;
}

// Draw model
void draw_model(torch::Tensor _data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi, std::string title="") {
	torch::Tensor x_model = torch::arange(0, 2, 0.01);
	torch::Tensor y_model = model(phi, x_model);

	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	matplot::axes_handle fx = F->nexttile();

	matplot::hold(fx, true);
	matplot::plot(fx, tensorTovector(_data[0].to(torch::kDouble)),
					  tensorTovector(_data[1].to(torch::kDouble)), "bo")->line_width(2);
	matplot::plot(fx, tensorTovector(x_model.to(torch::kDouble)),
					  tensorTovector(y_model.to(torch::kDouble)),"m-")->line_width(2);
	matplot::xlim(fx, {0,2});
	matplot::ylim(fx, {0,2});
	matplot::xlabel(fx, "x");
	matplot::ylabel(fx, "y");

    if( title != "" )
	    matplot::title(fx, title);
    matplot::show();
}

torch::Tensor compute_loss(torch::Tensor data_x, torch::Tensor data_y,
							torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi) {
    // write this function -- replace the line below
    // First make model predictions from data x
    // Then compute the squared difference between the predictions and true y values
    // Then sum them all and return
	//std::cout << "data_x: " << data_x.dtype() << " data_y: " << data_y.dtype() << " phi: " << phi.dtype() << '\n';
	torch::Tensor pred_y = model(phi, data_x);
	torch::Tensor loss = torch::sum(torch::pow((pred_y - data_y), 2));

    return loss;
}

void draw_loss_function(torch::Tensor (*compute_loss)(torch::Tensor, torch::Tensor,
		torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor),
		torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi_iters = torch::empty(0)) {

	// Make grid of intercept/slope values to plot
	torch::Tensor intercepts_mesh, slopes_mesh;
    std::vector<torch::Tensor> grids = torch::meshgrid({torch::arange(-1.0,1.0,0.002), torch::arange(0.0,2.0,0.02)}, "ij");
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
	F->size(800, 800);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	matplot::axes_handle fx = F->nexttile();
	matplot::hold(fx, true);

	//std::vector<double> lvls = linspace(-10.0, 10.0, 10);
	matplot::contour(fx, X, Y, Z)->line_width(2); //.levels(lvls);

    if( phi_iters.numel() > 0 ) {
	    matplot::plot(tensorTovector(phi_iters[0].to(torch::kDouble)),
			  	  	 tensorTovector(phi_iters[1].to(torch::kDouble)), "go-");
    }
	matplot::ylim(fx, {1, -1});
	matplot::xlabel(fx, "Intercept phi_0");
	matplot::ylabel(fx, "Slope, phi_1");
	matplot::show();
}

torch::Tensor compute_gradient(torch::Tensor data_x, torch::Tensor data_y, torch::Tensor phi) {
    // write this function, replacing the lines below
    float dl_dphi0 = torch::sum(2*(phi[0] + phi[1]*data_x - data_y)).data().item<float>();
    float dl_dphi1 = torch::sum(2*data_x*(phi[0] + phi[1]*data_x - data_y)).data().item<float>();

    // Return the gradient
    return torch::tensor({{dl_dphi0}, {dl_dphi1}});
}

torch::Tensor loss_function_1D(double dist_prop, torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor), torch::Tensor phi_start, torch::Tensor search_direction) {
    // Return the loss after moving this far
  return compute_loss(data[0], data[1], model, phi_start + search_direction * dist_prop);
}

float line_search(torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor),
		torch::Tensor phi, torch::Tensor gradient, float thresh=.00001, float max_dist = 0.1, int max_iter = 15, bool verbose=false) {
    // Initialize four points along the range we are going to search
    float a = 0;
    float b = 0.33 * max_dist;
    float c = 0.66 * max_dist;
    float d = 1.0 * max_dist;
    int n_iter = 0;

    // While we haven't found the minimum closely enough
    while( std::abs(b-c) > thresh && n_iter < max_iter ) {
        // Increment iteration counter (just to prevent an infinite loop)
        n_iter = n_iter+1;
        // Calculate all four points
        float lossa = loss_function_1D(a, data, model, phi, gradient).data().item<float>();
        float lossb = loss_function_1D(b, data, model, phi, gradient).data().item<float>();
        float lossc = loss_function_1D(c, data, model, phi, gradient).data().item<float>();
        float lossd = loss_function_1D(d, data, model, phi, gradient).data().item<float>();

        if(verbose) {
          printf("Iter %d, a=%3.3f, b=%3.3f, c=%3.3f, d=%3.3f\n", n_iter, a, b, c, d);
          printf("a %f, b%f, c%f, d%f\n", lossa, lossb, lossc, lossd);
        }

        // Rule #1 If point A is less than points B, C, and D then halve points B,C, and D
        if( lossa < lossb && lossa < lossc && lossa < lossd ) {
            b = b/2;
            c = c/2;
            d = d/2;
            //continue;
        }

        // Rule #2 If point b is less than point c then
        //                     point d becomes point c, and
        //                     point b becomes 1/3 between a and new d
        //                     point c becomes 2/3 between a and new d
        if( lossb < lossc ) {
            d = c;
            b = a + (d-a)/3;
            c = a + 2*(d-a)/3;
            //continue;
        }

        // Rule #2 If point c is less than point b then
        //                     point a becomes point b, and
        //                     point b becomes 1/3 between new a and d
        //                     point c becomes 2/3 between new a and d
        if( lossc < lossb ) {
        	a = b;
        	b = a + (d-a)/3;
        	c = a + 2*(d-a)/3;
        }
    }

    // Return average of two middle points
    return (b + c)/2.0;
}

torch::Tensor gradient_descent_step(torch::Tensor phi, torch::Tensor data, torch::Tensor (*model)(torch::Tensor, torch::Tensor)) {
    // update Phi with the gradient descent step (equation 6.3)
    // 1. Compute the gradient (you wrote this function above)
    // 2. Find the best step size alpha using line search function (above) -- use negative gradient as going downhill
    // 3. Update the parameters phi based on the gradient and the step size alpha.

	torch::Tensor gradient = compute_gradient(data[0], data[1], phi);
	float alpha = line_search(data, model, phi, gradient);
	torch::Tensor new_phi = phi - alpha * gradient;
    return new_phi;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = true;

	// Let's create our training data 12 pairs {x_i, y_i}
	// We'll try to fit the straight line model to these data
	torch::Tensor data = torch::tensor({{0.03,0.19,0.34,0.46,0.78,0.81,1.08,1.18,1.39,1.60,1.65,1.90},
                 {0.67,0.85,1.05,1.00,1.40,1.50,1.30,1.54,1.55,1.68,1.73,1.60}});

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw this model\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Initialize the parameters to some arbitrary values and draw the model
	torch::Tensor phi = torch::zeros({2,1});
	phi[0][0] = 0.6;      // Intercept
	phi[1][0] = -0.2;     // Slope
	if(plt) draw_model(data, model, phi, "Initial parameters");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute loss\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	torch::Tensor loss = compute_loss( data[0], data[1], model, torch::tensor({{0.6},{-0.2}}));
	printf("Your loss = %3.3f, Correct loss = %3.3f\n", loss.data().item<float>(), 12.367);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw loss function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	draw_loss_function(compute_loss, data, model);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw loss function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Compute the gradient using your function
	torch::Tensor gradient = compute_gradient(data[0], data[1], phi);

	printf("Your gradients: (%3.3f, %3.3f)\n", gradient[0].data().item<float>(),gradient[1].data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Approximate the gradients with finite differences\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Approximate the gradients with finite differences
	float delta = 0.0001;
	torch::Tensor dl_dphi0_est = (compute_loss(data[0], data[1], model, phi + torch::tensor({{delta}, {0.0f}}).to(torch::kFloat32)) -
	                    compute_loss(data[0], data[1], model, phi))/delta;

	torch::Tensor dl_dphi1_est = (compute_loss(data[0], data[1], model, phi + torch::tensor({{0.0f}, {delta}}).to(torch::kFloat32)) -
	                    compute_loss(data[0], data[1], model, phi))/delta;

	printf("Approx gradients: (%3.3f, %3.3f)\n", dl_dphi0_est.data().item<float>(), dl_dphi1_est.data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Measure loss and draw initial model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// # Initialize the parameters and draw the model
	int n_steps = 10;
	torch::Tensor phi_all = torch::zeros({2,n_steps+1});
	phi_all[0][0] = 1.6;
	phi_all[1][0] = -0.5;

	// Measure loss and draw initial model
	loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), Slice(0, 1)})); //:,0:1])
	std::string tlt = "Initial parameters, Loss = " + std::to_string(loss.data().item<float>());
	draw_model(data, model, phi_all.index({Slice(), Slice(0, 1)}), tlt);

	// Repeatedly take gradient descent steps
	for(auto& c_step : range (n_steps, 0)) {
	    // Do gradient descent step
	    torch::Tensor t = gradient_descent_step(phi_all.index({Slice(), Slice(c_step, c_step+1)}), data, model);
	    phi_all.index_put_({Slice(), Slice(c_step+1, c_step+2)}, t);
	    // Measure loss and draw model
	    loss =  compute_loss(data[0], data[1], model, phi_all.index({Slice(), Slice(c_step+1, c_step+2)}));

	    std::string tlt = "Iteration " + std::to_string(c_step+1) + ", loss = " + std::to_string(loss.data().item<float>());
	    draw_model(data, model, phi_all.index({Slice(), c_step+1}), tlt);
	}

	// Draw the trajectory on the loss function
	draw_loss_function(compute_loss, data, model, phi_all);

	std::cout << "Done!\n";
	return 0;
}


