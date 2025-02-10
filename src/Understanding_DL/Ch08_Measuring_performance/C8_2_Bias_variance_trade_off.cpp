/*
 * C8_2_Bias_variance_trade_off.cpp
 *
 *  Created on: Dec 21, 2024
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

// The true function that we are trying to estimate, defined on [0,1]
torch::Tensor true_function(torch::Tensor x) {
	torch::Tensor y = torch::exp(torch::sin(x*(2*3.1413)));
    return y;
}

// Generate some data points with or without noise
std::tuple<torch::Tensor , torch::Tensor> generate_data(int n_data, float sigma_y=0.3) {
    // Generate x values quasi uniformly
	torch::Tensor x = torch::ones({n_data});
    for(auto& i : range(n_data, 0)) {
    	x[i] = torch::uniform(x[i], i*1.0/n_data, (i+1)*1.0/n_data);
    }

    // y value from running through function and adding noise
    torch::Tensor y = torch::ones({n_data});
    for(auto& i : range(n_data, 0)) {
        y[i] = true_function(x[i]);
        y[i] += torch::normal(0, sigma_y, {1}).data().item<float>();
    }
    return std::make_tuple(x, y);
}

// Draw the fitted function, together with uncertainty used to generate points
void plot_function(torch::Tensor x_func, torch::Tensor y_func, torch::Tensor x_data=torch::empty(0),
		torch::Tensor y_data=torch::empty(0), torch::Tensor x_model=torch::empty(0), torch::Tensor y_model=torch::empty(0),
		torch::Tensor sigma_func=torch::empty(0), torch::Tensor sigma_model=torch::empty(0), std::string tlt="") {

    if( sigma_model.numel() > 0 ) {
       	auto F = figure(true);
        F->size(1200, 600);
        F->reactive_mode(false);
        F->tiledlayout(1, 2);
        F->position(0, 0);
    	std::string color = "b";
    	matplot::vector_1d x, y;
    	std::vector<double> x_f, x_m, y_f, y_m;
    	y_f = tensorTovector(y_func);
    	y_m = tensorTovector(y_model);
    	x_f = tensorTovector(x_func);
        x_m = tensorTovector(x_model);

    	auto ax = F->nexttile();
    	matplot::hold(ax, true);
        matplot::plot(ax, x_f, y_f, "k-")->line_width(2);
        matplot::plot(ax, x_m, y_m, "c--")->line_width(4);
        for(auto& i : range(static_cast<int>(x_m.size()) - 1, 1)) {
            double yf = y_f[i], ym = y_m[i];
            matplot::vector_1d x, y;
            x.push_back(x_m[i-1]);
            x.push_back(x_m[i]);
            x.push_back(x_m[i]);
            x.push_back(x_m[i-1]);
            if( ym > yf) {
            	y.push_back(y_f[i-1]);
            	y.push_back(y_f[i-1]);
            	y.push_back(ym);
            	y.push_back(ym);
            } else {
				y.push_back(y_m[i-1]);
				y.push_back(y_m[i-1]);
				y.push_back(yf);
				y.push_back(yf);
            }
            matplot::fill(x, y, color);
        }
    	matplot::xlim(ax, {0, 1});
        matplot::xlabel(ax, "Input, x");
        matplot::ylabel(ax, "Output, y");
        matplot::title(ax, tlt + " bias");

        auto bx = F->nexttile();
        matplot::hold(bx, true);
        matplot::plot(bx, tensorTovector(x_func), tensorTovector(y_func), "k-")->line_width(2);
        matplot::plot(bx, tensorTovector(x_model), tensorTovector(y_model), "c--")->line_width(3);
    	matplot::plot(bx, tensorTovector(x_model),
    			tensorTovector(y_model.add(-2*sigma_model)), "r-.")->line_width(1);
    	matplot::plot(bx, tensorTovector(x_model),
    			tensorTovector(y_model.add(2*sigma_model)), "r-.")->line_width(1);
    	matplot::xlim(bx, {0, 1});
        matplot::xlabel(bx, "Input, x");
        matplot::ylabel(bx, "Output, y");
        matplot::title(bx, tlt + " variance");
        matplot::show();
    } else {
    	auto F = figure(true);
    	F->size(800, 600);
    	F->reactive_mode(false);
    	F->tiledlayout(1, 1);
    	F->position(0, 0);

    	auto ax = F->nexttile();
    	matplot::hold(ax, true);
        matplot::plot(ax, tensorTovector(x_func), tensorTovector(y_func), "k-")->line_width(2);

        if(sigma_func.numel() > 0) {
        	matplot::plot(ax, tensorTovector(x_func),
        			tensorTovector(y_func.add(-2*sigma_func.data().item<double>())), "b-:")->line_width(1);
        	matplot::plot(ax, tensorTovector(x_func),
        			tensorTovector(y_func.add(2*sigma_func.data().item<double>())), "b-:")->line_width(1);
        }

        if( x_data.numel() > 0 ) {
        	matplot::plot(ax, tensorTovector(x_data), tensorTovector(y_data), "mo")->line_width(2);;
        }

        if( x_model.numel() > 0 ) {
        	matplot::plot(ax, tensorTovector(x_model), tensorTovector(y_model), "c--")->line_width(3);
        }
    	matplot::xlim(ax, {0, 1});
        matplot::xlabel(ax, "Input, x");
        matplot::ylabel(ax, "Output, y");
        matplot::show();
    }
}

// Define model -- beta is a scalar and omega has size n_hidden,1
torch::Tensor _network(torch::Tensor x, torch::Tensor beta, torch::Tensor omega) {
    // Retrieve number of hidden units
    int n_hidden = omega.size(0);

    torch::Tensor y = torch::zeros_like(x);
    for(auto& c_hidden : range(n_hidden, 0)) {
        // Evaluate activations based on shifted lines (figure 8.4b-d)
    	torch::Tensor line_vals =  x  - c_hidden*1.0/n_hidden;
    	torch::Tensor h =  line_vals * (line_vals > 0).to(torch::kInt);

        // Weight activations by omega parameters and sum
        y = y + omega[c_hidden] * h;
    }
    // Add bias, beta
    y = y.add(beta);
    return y;
}

// This fits the n_hidden+1 parameters (see fig 8.4a) in closed form.
// If you have studied linear algebra, then you will know it is a least
// squares solution of the form (A^TA)^-1A^Tb.  If you don't recognize that,
// then just take it on trust that this gives you the best possible solution.
std::tuple<torch::Tensor, torch::Tensor> fit_model_closed_form(torch::Tensor x, torch::Tensor y, int n_hidden) {
	int n_data = x.size(0);
	torch::Tensor A = torch::ones({n_data, n_hidden+1});

	for(auto& i : range(n_data, 0)) {
	    for(auto& j : range(n_hidden, 1)) {
	        A[i][j] = (x[i]-(j-1.0)/n_hidden).data().item<float>();
	        if(A[i][j].data().item<float>() < 0)
	            A[i][j] = 0;
	    }
	}

	torch::Tensor beta_omega, t_1, t_2, t_3;
	std::optional<double> rc = {};
	std::optional<c10::basic_string_view<char>>  d = {"gelsy"}; // 'gelsy’ 用于 CPU 输入， ‘gels’ 用于 CUDA 输入
	std::tie(beta_omega, t_1, t_2, t_3) = torch::linalg_lstsq(A, y, rc, d); //[0]

	torch::Tensor beta = beta_omega[0];
	torch::Tensor omega = beta_omega.index({Slice(1, None)});

	return std::make_tuple(beta, omega);
}

// Run the model many times with different datasets and return the mean and variance
std::tuple<torch::Tensor, torch::Tensor> get_model_mean_variance( torch::Tensor x_model,
		int n_data, int n_datasets, int n_hidden, float sigma_func) {

    // Create array that stores model results in rows
	torch::Tensor y_model_all = torch::zeros({n_datasets, x_model.size(0)});

    for(auto& c_dataset : range(n_datasets, 0)) {
        //  Generate n_data x,y, pairs with standard deviation sigma_func
	    torch::Tensor x_data, y_data;
	    std::tie(x_data, y_data) = generate_data(n_data, sigma_func);

        // -- Fit the model
	    torch::Tensor beta, omega;
	    std::tie(beta, omega) = fit_model_closed_form(x_data, y_data, n_hidden);

        //-- Run the fitted model on x_model
	    torch::Tensor y_model = _network(x_model, beta, omega);

        // Store the model results
        y_model_all.index_put_({c_dataset,Slice()}, y_model);
    }
    // Get mean and standard deviation of model
    c10::OptionalArrayRef<long int> dim = {0};
    torch::Tensor mean_model = torch::mean(y_model_all, dim);
    torch::Tensor std_model = torch::std(y_model_all, dim);

    // Return the mean and standard deviation of the fitted model
    return std::make_tuple(mean_model, std_model);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(345);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Generate true function\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Generate true function
	torch::Tensor x_func = torch::linspace(0, 1.0, 100);
	torch::Tensor y_func = true_function(x_func);
	std::cout << y_func << '\n';


	// Generate some data points
	float sigma_func = 0.3;
	int n_data = 15;
	torch::Tensor x_data,y_data;
	std::tie(x_data, y_data) = generate_data(n_data, sigma_func);
	std::cout << x_data << '\n';
	std::cout << y_data << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the functinon, data and uncertainty\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Plot the functinon, data and uncertainty
	plot_function(x_func.to(torch::kDouble), y_func.to(torch::kDouble), x_data.to(torch::kDouble),
			y_data.to(torch::kDouble), torch::empty(0), torch::empty(0),
			torch::tensor({sigma_func}).to(torch::kDouble));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Closed form solution\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	int n_hidden=3;
	torch::Tensor beta, omega;
	// Closed form solution
	std::tie(beta, omega) = fit_model_closed_form(x_data, y_data, n_hidden);
	std::cout << "beta:\n" << beta << "\nomega:\n" << omega << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Get prediction for model across graph range\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Get prediction for model across graph range
	torch::Tensor x_model = torch::linspace(0, 1, 100);
	torch::Tensor y_model = _network(x_model, beta, omega);

	// Draw the function and the model
	plot_function(x_func.to(torch::kDouble), y_func.to(torch::kDouble), x_data.to(torch::kDouble),
			y_data.to(torch::kDouble), x_model.to(torch::kDouble), y_model.to(torch::kDouble));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the noise, bias and variance as a function of capacity\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::vector<int> hidden_variables = {1,2,3,4,5,6,7,8,9,10,11,12};
	torch::Tensor bias = torch::zeros({static_cast<int64_t>(hidden_variables.size()), 1});
	torch::Tensor variance = torch::zeros({static_cast<int64_t>(hidden_variables.size()), 1});

	int n_datasets = 100;
	n_data = 15;
	sigma_func = 0.3;
	std::vector<double> xx;

	for(auto& c_hidden : hidden_variables) {
		// Get mean and variance of fitted model
		torch::Tensor mean_model, std_model;
		std::tie(mean_model, std_model) = get_model_mean_variance(x_model, n_data, n_datasets, c_hidden, sigma_func);

		variance[c_hidden-1] = torch::mean(std_model).data().item<double>();
		// Compute bias (average squared deviation of mean fitted model around true function)
		bias[c_hidden-1] = torch::mean(mean_model - y_func).data().item<double>();

		if(c_hidden == 4 || c_hidden == 8 || c_hidden == 12) {
			plot_function(x_func.to(torch::kDouble), y_func.to(torch::kDouble), torch::empty(0), torch::empty(0),
					    		  x_model.to(torch::kDouble), mean_model.to(torch::kDouble), torch::empty(0),
								  std_model.to(torch::kDouble), std::to_string(c_hidden) + " regions");
		}
		xx.push_back(c_hidden*1.0);
	}

	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);
	matplot::plot(fx, xx, tensorTovector(variance.to(torch::kDouble)),
			"-:")->line_width(2).display_name("Variance");
	matplot::plot(fx, xx, tensorTovector(bias.to(torch::kDouble)),
			"-")->line_width(2).display_name("Bias");
	matplot::xlabel("Model capacity");
	matplot::ylabel("Variance");
	matplot::legend(fx, {});
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





