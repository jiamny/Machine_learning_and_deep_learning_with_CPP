/*
 * C5_2_Binary_Cross_Entropy_Loss.cpp
 *
 *  Created on: Sep 23, 2024
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


// Define a shallow neural network
torch::Tensor shallow_nn(torch::Tensor x, torch::Tensor beta_0, torch::Tensor omega_0, torch::Tensor beta_1, torch::Tensor omega_1) {
    // Make sure that input data is (1 x n_data) array
    int n_data = x.size(0);
    x = torch::reshape(x,{1, n_data});

    _ReLU relu = _ReLU();
    // This runs the network for ALL of the inputs, x at once so we can draw graph
    torch::Tensor h1 = relu.forward(torch::matmul(beta_0, torch::ones({1,n_data})) + torch::matmul(omega_0,x));

    torch::Tensor model_out = torch::matmul(beta_1, torch::ones({1,n_data})) + torch::matmul(omega_1, h1);
    return model_out;
}

// Get parameters for model -- we can call this function to easily reset them
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_parameters() {
    // And we'll create a network that approximately fits it
	torch::Tensor beta_0 = torch::zeros({3,1});  // formerly theta_x0
	torch::Tensor omega_0 = torch::zeros({3,1}); // formerly theta_x1
	torch::Tensor beta_1 = torch::zeros({1,1});  // formerly phi_0
	torch::Tensor omega_1 = torch::zeros({1,3}); // formerly phi_x

    beta_0[0][0] = 0.3; beta_0[1][0] = -1.0; beta_0[2][0] = -0.5;
    omega_0[0][0] = -1.0; omega_0[1][0] = 1.8; omega_0[2][0] = 0.65;
    beta_1[0][0] = 2.6;
    omega_1[0][0] = -24.0; omega_1[0][1] = -8.0; omega_1[0][2] = 50.0;

    return std::make_tuple(beta_0, omega_0, beta_1, omega_1);
}


// Utility function for plotting data
void plot_binary_classification(torch::Tensor x_model, torch::Tensor out_model, torch::Tensor lambda_model,
		torch::Tensor x_data = torch::empty(0), torch::Tensor y_data = torch::empty(0), std::string title="") {
    // Make sure model data are 1D arrays
    x_model = torch::squeeze(x_model);
    out_model = torch::squeeze(out_model);
    lambda_model = torch::squeeze(lambda_model);

	matplot::figure(true)->size(800, 600);
	matplot::subplot(1, 2, 0);

	matplot::plot(tensorTovector(x_model.to(torch::kDouble)), tensorTovector(out_model.to(torch::kDouble)), "r-:")->line_width(2);
	matplot::xlabel("Input, x");
	matplot::ylabel("Model output");
	matplot::xlim({0,1});
	matplot::ylim({-4,4});
    if( title != "")
	    matplot::title(title);

    matplot::subplot(1, 2, 1);
    matplot::plot(tensorTovector(x_model.to(torch::kDouble)), tensorTovector(lambda_model.to(torch::kDouble)), "b--")->line_width(2);
    matplot::xlabel("Input, x");
    matplot::ylabel("lambda or Pr(y=1|x)");
    matplot::xlim({0,1});
    matplot::ylim({-0.05,1.05});
    if(title != "" )
	    matplot::title(title);

    if(x_data.numel() > 0) {
    	matplot::hold(true);
	    matplot::plot(tensorTovector(x_data.to(torch::kDouble)), tensorTovector(y_data.to(torch::kDouble)), "mo");
    }
    matplot::show();
}

// Return probability under Bernoulli distribution for observed class y
torch::Tensor Bernoulli_distribution(torch::Tensor y, torch::Tensor lambda_param) {
    // write in the equation for the Bernoulli distribution
    // Equation 5.17 from the notes (you will need np.power)

	torch::Tensor prob = torch::pow((1. - lambda_param), (1. - y)) * torch::pow(lambda_param, y);

    return prob;
}

// Return the likelihood of all of the data under the model
torch::Tensor compute_likelihood(torch::Tensor y_train, torch::Tensor lambda_param) {
    // compute the likelihood of the data -- the product of the Bernoulli probabilities for each data point
    // Top line of equation 5.3 in the notes
    // You will need np.prod() and the bernoulli_distribution function you used above

	torch::Tensor P = Bernoulli_distribution(y_train, lambda_param);
	torch::Tensor likelihood = torch::prod(P);

  return likelihood;
}

//  Return the negative log likelihood of the data under the model
torch::Tensor compute_negative_log_likelihood(torch::Tensor y_train, torch::Tensor lambda_param) {
    // compute the likelihood of the data -- don't use the likelihood function above -- compute the negative sum of the log probabilities
    // You will need np.sum(), np.log()
	torch::Tensor t = Bernoulli_distribution(y_train, lambda_param);
	torch::Tensor P = torch::log(t);
	torch::Tensor nlls = -1.0*torch::sum(P);

    return nlls;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = true;

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Binary classification\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Let's create some 1D training data
	torch::Tensor  x_train = torch::tensor({0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,
                   0.87168699,0.58858043});
	torch::Tensor  y_train = torch::tensor({0,1,1,0,0,1,
                    1,0,0,1,0,1,
                    0,1,1,0,1,0,
                    1,1});

	// Get parameters for the model
	torch::Tensor beta_0, omega_0, beta_1, omega_1;
	std::tie(beta_0, omega_0, beta_1, omega_1)= get_parameters();

	// Define a range of input values
	torch::Tensor x_model = torch::arange(0,1,0.01);
	// Run the model to get values to plot and plot it.
	torch::Tensor model_out= shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	torch::Tensor lambda_model = Sigmoid(model_out);

	if(plt) plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's double check bernoulli distribution\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	printf("Correct answer = %3.3f, Your answer = %3.3f\n", 0.8,
			Bernoulli_distribution(torch::tensor({0.}), torch::tensor({0.2})).data().item<float>());
	printf("Correct answer = %3.3f, Your answer = %3.3f\n", 0.2,
			Bernoulli_distribution(torch::tensor({1.}), torch::tensor({0.2})).data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's double check compute likelihood\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Let's test this
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	// Use our neural network to predict the Bernoulli parameter lambda
	model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	torch::Tensor lambda_train = Sigmoid(model_out);

	std::cout << lambda_train << '\n';

	// Compute the likelihood
	torch::Tensor likelihood = compute_likelihood(y_train, lambda_train);
	// Let's double check we get the right answer before proceeding
	printf("Correct answer = %9.9f, Your answer = %9.9f\n", 0.000070237, likelihood.data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's double check compute the log likelihood\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	//  Let's test this
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	// Use our neural network to predict the mean of the Gaussian
	model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	// Pass through the sigmoid function
	lambda_train = Sigmoid(model_out);
	// Compute the log likelihood
	torch::Tensor  nll = compute_negative_log_likelihood(y_train, lambda_train);
	// Let's double check we get the right answer before proceeding
	printf("Correct answer = %9.9f, Your answer = %9.9f\n", 9.563639387, nll.data().item<float>());


	// Define a range of values for the parameter
	torch::Tensor beta_1_vals = torch::arange(-2,6.0,0.1);
	// Create some arrays to store the likelihoods, negative log likelihoods
	// Define a range of values for the parameter
	torch::Tensor likelihoods = torch::zeros_like(beta_1_vals);
	// Define a range of values for the parameter
	torch::Tensor nlls = torch::zeros_like(beta_1_vals);

	// Initialise the parameters
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	for(auto& count : range(static_cast<int>(beta_1_vals.size(0)), 0)) {
	    // Set the value for the parameter
	    beta_1[0][0] = beta_1_vals[count];
	    // Run the network with new parameters
	    model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	    lambda_train = Sigmoid(model_out);
	    // Compute and store the two values
	    likelihoods[count] = compute_likelihood(y_train,lambda_train);
	    nlls[count] = compute_negative_log_likelihood(y_train, lambda_train);
	    // Draw the model for every 20th parameter setting
	    if( count % 20 == 0 ) {
	    	// Run the model to get values to plot and plot it.
	    	model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	    	lambda_model = Sigmoid(model_out);
	    	std::string title = "beta-1[0] = " + to_string_with_precision(beta_1[0][0].data().item<float>(), 3);
	    	if(plt) plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train, title);
	    }
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's plot the likelihood and negative log likelihood as a function of the value of the offset beta1\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// # Now let's plot the likelihood and negative log likelihood as a function of the value of the offset beta1
	matplot::figure(true)->size(800, 600);
	matplot::subplot(1, 2, 0);

	float max_lk = beta_1_vals[torch::argmax(likelihoods).data().item<int>()].data().item<float>();
	std::vector<double> max_lk_x;
	for(auto& i : range(static_cast<int>(likelihoods.size(0)), 0))
		max_lk_x.push_back(max_lk);

	matplot::hold(true);
	matplot::plot(tensorTovector(beta_1_vals.to(torch::kDouble)), tensorTovector(likelihoods.to(torch::kDouble)), "r-")->line_width(2);
	matplot::plot(max_lk_x, tensorTovector(likelihoods.to(torch::kDouble)), "b-:")->line_width(1);
	matplot::xlabel("beta-1[0]");
	matplot::ylabel("likelihood");

	matplot::subplot(1, 2, 1);
	float min_nll = beta_1_vals[torch::argmin(nlls).data().item<int>()].data().item<float>();
	std::vector<double> min_nll_x;
	for(auto& i : range(static_cast<int>(nlls.size(0)), 0))
		min_nll_x.push_back(min_nll);

	matplot::hold(true);
	matplot::plot(tensorTovector(beta_1_vals.to(torch::kDouble)), tensorTovector(nlls.to(torch::kDouble)), "m-.")->line_width(2);
	matplot::plot(min_nll_x, tensorTovector(nlls.to(torch::kDouble)), "b-:")->line_width(1);
	matplot::xlabel("beta-1[0]");
	matplot::ylabel("negative log likelihood");

	matplot::show();
	// Hopefully, you can see that the maximum of the likelihood fn is at the same position as the minimum negative log likelihood
	// Let's check that:
	printf("Maximum likelihood = %f, at beta_1 = %3.3f\n",
			likelihoods[torch::argmax(likelihoods).data().item<int>()].data().item<float>(),
			beta_1_vals[torch::argmax(likelihoods).data().item<int>()].data().item<float>());
	printf("Minimum negative log likelihood = %f, at beta_1 = %3.3f\n",
			nlls[torch::argmin(nlls).data().item<int>()].data().item<float>(),
			beta_1_vals[torch::argmin(nlls).data().item<int>()].data().item<float>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the best model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Plot the best model
	beta_1[0][0] = beta_1_vals[torch::argmin(nlls).data().item<int>()];
	model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	lambda_model = Sigmoid(model_out);
	std::string title = "beta-1[0] = " + to_string_with_precision(beta_1[0][0].data().item<float>(), 3);
	plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train, title);


	std::cout << "Done!\n";
	return 0;
}





