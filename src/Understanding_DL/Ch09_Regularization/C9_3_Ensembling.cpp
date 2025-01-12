/*
 * C9_3_Ensembling.cpp
 *
 *  Created on: Jan 10, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
//#include "../../Utils/UDL_util.h"
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
std::tuple<torch::Tensor, torch::Tensor> generate_data(int n_data, float sigma_y=0.3) {
    // Generate x values quasi uniformly
	torch::Tensor x = torch::ones({n_data});
    for(auto& i : range(n_data, 0)) {
        x[i].uniform_(i*1.0/n_data, (i+1)*1.0/n_data);// = torch::randn(i/n_data, (i+1)/n_data, 1)
    }

    // y value from running through function and adding noise
    torch::Tensor y = torch::ones({n_data});
    for(auto& i : range(n_data, 0)) {
        y[i] = true_function(x[i]).data().item<float>();
        y[i] += torch::normal(0, sigma_y, {1}).data().item<float>();
    }
    return std::make_tuple(x, y);
}

// Draw the fitted function, together with uncertainty used to generate points
void plot_function(torch::Tensor x_func, torch::Tensor y_func, torch::Tensor x_data=torch::empty(0),
		torch::Tensor y_data=torch::empty(0), torch::Tensor sigma_func=torch::empty(0), std::string tlt="",
		torch::Tensor x_model=torch::empty(0), torch::Tensor y_model=torch::empty(0),
		torch::Tensor sigma_model=torch::empty(0)) {

    auto F = matplot::figure(true);
    F->size(800, 600);
    F->reactive_mode(false);
    F->tiledlayout(1, 1);
    F->position(0, 0);

    auto fx = F->nexttile();
    matplot::hold(fx, true);
    matplot::plot(fx, tensorTovector(x_func.to(torch::kDouble)),
    		tensorTovector(y_func.to(torch::kDouble)), "k-")->line_width(2);

    if( sigma_func.numel() > 0) {
    	std::vector<double> xx, yU, yL;
    	float sigma = sigma_func.data().item<float>();
    	for(int i = 0; i < x_func.size(0); i++) {
    		xx.push_back(x_func[i].data().item<float>());
    		yL.push_back(y_func[i].data().item<float>() - 2 * sigma);
    		yU.push_back(y_func[i].data().item<float>() + 2 * sigma);
    	}
    	matplot::plot(fx, xx, yU, "g--")->line_width(2);
    	matplot::plot(fx, xx, yL, "g--")->line_width(2);
      //ax.fill_between(x_func, y_func-2*sigma_func, y_func+2*sigma_func, color='lightgray')
    }

    if( x_data.numel() > 0) {
    	matplot::plot(fx, tensorTovector(x_data.to(torch::kDouble)),
    			tensorTovector(y_data.to(torch::kDouble)), "mo");
    }

    if( x_model.numel() > 0) {
    	matplot::plot(fx, tensorTovector(x_model.to(torch::kDouble)),
    			tensorTovector(y_model.to(torch::kDouble)), "c-")->line_width(2);
    }

    if( sigma_model.numel() > 0) {
      //ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')
    }

    matplot::xlim(fx, {0,1});
    matplot::xlabel(fx, "Input, x");
    matplot::ylabel(fx, "Output, y");
    if(tlt != "")
    	matplot::title(fx, tlt);

    matplot::show();
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
    y = y + beta;

    return y;
}

/*
 This fits the n_hidden+1 parameters (see fig 8.4a) in closed form.
 If you have studied linear algebra, then you will know it is a least
 squares solution of the form (A^TA)^-1A^Tb.  If you don't recognize that,
 then just take it on trust that this gives you the best possible solution.
 */
std::tuple<torch::Tensor, torch::Tensor> fit_model_closed_form(torch::Tensor x, torch::Tensor y, int n_hidden) {
  int n_data = x.size(0);
  torch::Tensor A = torch::ones({n_data, n_hidden+1});
  for(auto& i : range(n_data, 0)) {
      for(auto& j : range(n_hidden, 1)) {
          // Compute preactivation
          A[i][j] = x[i].data().item<float>() - (j-1)*1.0/n_hidden;
          // Apply the ReLU function
          if( A[i][j].data().item<float>() < 0)
              A[i][j] = 0;
      }
  }
  // Add a tiny bit of regularization
  float reg_value = 0.00001;
  torch::Tensor regMat = reg_value * torch::eye(n_hidden+1);
  regMat[0][0] = 0;

  torch::Tensor ATA = torch::matmul(A.t(), A) + regMat;
  torch::Tensor ATAInv = torch::linalg::inv(ATA);
  torch::Tensor ATAInvAT = torch::matmul(ATAInv, A.t());
  torch::Tensor beta_omega = torch::matmul(ATAInvAT, y);
  torch::Tensor beta = beta_omega[0];
  torch::Tensor omega = beta_omega.index({Slice(1, None)});

  return std::make_tuple(beta, omega);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the function, data and uncertainty\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Generate true function
	torch::Tensor x_func = torch::linspace(0, 1.0, 100);
	torch::Tensor y_func = true_function(x_func);

	// Generate some data points
	float sigma_func = 0.3;
	int n_data = 30;
	torch::Tensor x_data, y_data;
	std::tie(x_data, y_data) = generate_data(n_data, sigma_func);

	// Plot the function, data and uncertainty
	plot_function(x_func, y_func, x_data, y_data, torch::tensor({sigma_func}));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Fits the n_hidden+1 parameters\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Closed form solution
	torch::Tensor beta, omega;
	std::tie(beta, omega) = fit_model_closed_form(x_data,y_data, 14);

	// Get prediction for model across graph range
	torch::Tensor x_model = torch::linspace(0, 1, 100);
	torch::Tensor y_model = _network(x_model, beta, omega);

	// Draw the function and the model
	plot_function(x_func, y_func, x_data,y_data, torch::empty(0), "", x_model, y_model);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Closed form solution\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Compute the mean squared error between the fitted model (cyan) and the true curve (black)
	torch::Tensor mean_sq_error = torch::mean((y_model-y_func) * (y_model-y_func));
	printf("Mean square error = %3.3f\n", mean_sq_error.data().item<float>());

	// Now let's resample the data with replacement four times.
	int n_model = 4;
	// Array to store the prediction from all of our models
	torch::Tensor all_y_model = torch::zeros({n_model, y_model.size(0)});

	// For each model
	for(auto& c_model : range(n_model, 0)) {
	    // Sample data indices with replacement (use np.random.choice)

		torch::Tensor resampled_indices = torch::randint(0, n_data,{25});

	    // Extract the resampled x and y data
		torch::Tensor x_data_resampled = x_data.index_select(0, resampled_indices);
		torch::Tensor y_data_resampled = y_data.index_select(0, resampled_indices);

	    // Fit the model
	    std::tie(beta, omega) = fit_model_closed_form(x_data_resampled, y_data_resampled, 14);

	    // Run the model
		torch::Tensor y_model_resampled = _network(x_model, beta, omega);

	    // Store the results
	    all_y_model.index_put_({c_model, Slice()}, y_model_resampled);

	    // Draw the function and the model
	    std::string tlt = "model " + std::to_string(c_model + 1);
	    plot_function(x_func, y_func, x_data, y_data, torch::empty(0), tlt, x_model, y_model_resampled);

	    // Compute the mean squared error between the fitted model (cyan) and the true curve (black)
	    mean_sq_error = torch::mean((y_model_resampled-y_func) * (y_model_resampled-y_func));
	    printf("Mean square error = %3.3f\n", mean_sq_error.data().item<float>());
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the median of the results\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Plot the median of the results
	// find the median prediction
	torch::Tensor y_model_median, _;
	std::tie(y_model_median, _)= torch::median(all_y_model, 0, false); //all_y_model[0,:]

	// Draw the function and the model
	std::string tlt = "model median";
	plot_function(x_func, y_func, x_data,y_data, torch::empty(0), tlt, x_model, y_model_median);

	// Compute the mean squared error between the fitted model (cyan) and the true curve (black)
	mean_sq_error = torch::mean((y_model_median-y_func) * (y_model_median-y_func));
	printf("Mean square error = %3.3f\n", mean_sq_error.data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the mean of the results\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	c10::OptionalArrayRef<long int> d = {0};
	torch::Tensor y_model_mean = torch::mean(all_y_model, d);

	// Draw the function and the model
	tlt = "model mean";
	plot_function(x_func, y_func, x_data, y_data, torch::empty(0), tlt, x_model, y_model_mean);

	// Compute the mean squared error between the fitted model (cyan) and the true curve (black)
	mean_sq_error = torch::mean((y_model_mean-y_func) * (y_model_mean-y_func));
	printf("Mean square error = %3.3f\n", mean_sq_error.data().item<float>());

	std::cout << "Done!\n";
	return 0;
}



