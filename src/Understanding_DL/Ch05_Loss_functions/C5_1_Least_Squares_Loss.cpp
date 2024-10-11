/*
 * C5_1_Least_Squares_Loss.cpp
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
torch::Tensor  shallow_nn(torch::Tensor x, torch::Tensor beta_0, torch::Tensor omega_0, torch::Tensor beta_1, torch::Tensor omega_1) {
    // Make sure that input data is (1 x n_data) array
    int n_data = x.size(0);
    x = x.reshape({1, n_data});

    _ReLU relu = _ReLU();
    // This runs the network for ALL of the inputs, x at once so we can draw graph
    torch::Tensor h1 = relu.forward(torch::matmul(beta_0, torch::ones({1,n_data}).to(x.dtype())) + torch::matmul(omega_0,x));
    torch::Tensor y = torch::matmul(beta_1, torch::ones({1, n_data}).to(x.dtype())) + torch::matmul(omega_1, h1);
    return y;
}

// Get parameters for model -- we can call this function to easily reset them
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_parameters() {
	// And we'll create a network that approximately fits it
	torch::Tensor beta_0 = torch::zeros({3,1}).to(torch::kDouble);		// formerly theta_x0
	torch::Tensor omega_0 = torch::zeros({3,1}).to(torch::kDouble); 	//  formerly theta_x1
	torch::Tensor beta_1 = torch::zeros({1,1}).to(torch::kDouble);  	// formerly phi_0
	torch::Tensor omega_1 = torch::zeros({1,3}).to(torch::kDouble); 	// formerly phi_x

	beta_0[0][0] = 0.3; beta_0[1][0] = -1.0; beta_0[2][0] = -0.5;
	omega_0[0][0] = -1.0; omega_0[1][0] = 1.8; omega_0[2][0] = 0.65;
	beta_1[0][0] = 0.1;
	omega_1[0][0] = -2.0; omega_1[0][1] = -1.0; omega_1[0][2] = 7.0;

	return std::make_tuple(beta_0, omega_0, beta_1, omega_1);
}

// Utility function for plotting data
void plot_univariate_regression(torch::Tensor x_model, torch::Tensor y_model, torch::Tensor x_data = torch::empty(0),
		torch::Tensor y_data = torch::empty(0), double sigma_model = 0.0, std::string title= "") {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	matplot::axes_handle fx = F->nexttile();
	// Make sure model data are 1D arrays
	std::vector<double> x_mod = tensorTovector(torch::squeeze(x_model));
	std::vector<double> y_mod = tensorTovector(torch::squeeze(y_model));

	matplot::plot(fx, x_mod, y_mod, "b")->line_width(2);
	matplot::hold(fx, true);
	if( sigma_model !=  0. ) {
	//	matplot::fill(x_model, y_model-2*sigma_model, y_model+2*sigma_model);
		matplot::plot(fx, x_mod, tensorTovector(torch::squeeze(y_model)-2*sigma_model), "c")->line_width(2);
		matplot::plot(fx, x_mod, tensorTovector(torch::squeeze(y_model)+2*sigma_model), "c")->line_width(2);
	}

	if(title != "")
		matplot::title(title);

	if(x_data.numel() > 0 ) {
		std::vector<double> x_dt = tensorTovector(x_data);
		std::vector<double> y_dt = tensorTovector(y_data);
		matplot::plot(fx, x_dt, y_dt, "ko")->line_width(4);
	}
	matplot::xlabel(fx, "Input, x");
	matplot::ylabel(fx, "Output, y");
	matplot::xlim(fx, {0,1});
	matplot::ylim(fx, {-1,1});

	matplot::show();
}

// # Return probability under normal distribution
torch::Tensor Normal_distribution(torch::Tensor y, double mu, double sigma) {

    // Equation 5.7 from the notes (you will need np.sqrt() and np.exp(), and math.pi)
    // Don't use the numpy version -- that's cheating!
    // Replace the line below
	torch::Tensor prob = (1.0 / std::sqrt(2*M_PI*std::pow(sigma, 2))) * torch::exp(-1*torch::pow(y - mu, 2)/(2.*std::pow(sigma, 2)));

    return prob;
}

void plot_normal_distribution(torch::Tensor y_gauss, torch::Tensor gauss_prob, std::string tlt) {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	matplot::axes_handle fx = F->nexttile();

	matplot::plot(fx, tensorTovector(y_gauss), tensorTovector(gauss_prob), "r-")->line_width(2);
	matplot::xlabel(fx, "Input, y");
	matplot::ylabel(fx, "Probability Pr(y)");
	matplot::xlim(fx, {-5, 5});
	matplot::title(fx, tlt.c_str());
	matplot::show();
}

// Return the likelihood of all of the data under the model
torch::Tensor compute_likelihood(torch::Tensor y, torch::Tensor mu, double sigma) {
  // compute the likelihood of the data -- the product of the normal probabilities for each data point
  // Top line of equation 5.3 in the notes
  // You will need np.prod() and the normal_distribution function you used above

	torch::Tensor P = torch::zeros(y.size(0)).to(y.dtype());
	for(auto& i : range(static_cast<int>(y.size(0)), 0)) {
		torch::Tensor t = (1.0 / std::sqrt(2*M_PI*std::pow(sigma, 2))) * torch::exp(-1*torch::pow(y[i] - mu[i], 2)/(2.*std::pow(sigma, 2)));
		P[i] = t.data().item<double>();
	}
	torch::Tensor likelihood = torch::prod(P);

  return likelihood;
}

// Return the negative log likelihood of the data under the model
torch::Tensor compute_negative_log_likelihood(torch::Tensor y, torch::Tensor mu, double sigma) {
  // In other words, compute minus one times the sum of the log probabilities
  // Equation 5.4 in the notes
  // You will need np.sum(), np.log()

	torch::Tensor P = torch::zeros(y.size(0)).to(y.dtype());
	for(auto& i : range(static_cast<int>(y.size(0)), 0)) {
		torch::Tensor t = (1.0 / std::sqrt(2*M_PI*std::pow(sigma, 2))) * torch::exp(-1*torch::pow(y[i] - mu[i], 2)/(2.*std::pow(sigma, 2)));
		P[i] = torch::log(t).data().item<double>();
	}
	torch::Tensor nll = -1.0*torch::sum(P);

	return nll;
}

// Return the squared distance between the observed data (y_train) and the prediction of the model (y_pred)
torch::Tensor  compute_sum_of_squares(torch::Tensor y_train, torch::Tensor y_pred) {
	// Compute the sum of squared distances between the training data and the model prediction
	// Eqn 5.10 in the notes.  Make sure that you understand this, and ask questions if you don't

	torch::Tensor sum_of_squares = torch::sum(torch::square(y_train - y_pred));

	return sum_of_squares;
}

void plot_likelihood_negative_log_likelihood_and_least_squares( std::vector<double> beta_1_v,
		std::vector<double> likelihoods_v, std::vector<double> max_beta_x, std::vector<double> max_beta_y,
		std::vector<double> nlls_v, std::vector<double> min_beta_x, std::vector<double> min_beta_y,
		std::vector<double> sum_squares_v = {}, std::vector<double> min_beta_x2 = {}, std::vector<double> min_beta_y2 = {}, std::string xlab="sigma") {

	auto F = figure(true);
	if( sum_squares_v.size() > 0 ) {
		F->size(1500, 600);
		F->tiledlayout(1, 3);
	} else {
		F->size(1000, 600);
		F->tiledlayout(1, 2);
	}
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);

	auto fx1 = F->nexttile();
	matplot::hold(fx1, true);
	matplot::plot(fx1, beta_1_v, likelihoods_v, "r--")->line_width(2).display_name("likelihood");
	matplot::plot(fx1, max_beta_x, max_beta_y, "k-:")->line_width(2).display_name("Maximum likelihood");
	matplot::xlabel(fx1, xlab);
	matplot::ylabel(fx1, "likelihood");
	matplot::legend(fx1, {});

	auto fx2 = F->nexttile();
	matplot::hold(fx2, true);
	matplot::plot(fx2, beta_1_v, nlls_v, "b--")->line_width(2).display_name("negative log likelihood");
	matplot::plot(fx2, min_beta_x, min_beta_y, "k-:")->line_width(2).display_name("Minimum negative log likelihood");
	matplot::xlabel(fx2, xlab);
	matplot::ylabel(fx2, "negative log likelihood");
	matplot::legend(fx2, {});

	if( sum_squares_v.size() > 0 ) {
		auto fx3 = F->nexttile();
		matplot::hold(fx3, true);
		matplot::plot(fx3, beta_1_v, sum_squares_v, "m--")->line_width(2).display_name("sum of squares");
		matplot::plot(fx3, min_beta_x2, min_beta_y2, "k-:")->line_width(2).display_name("Least sum of squares");
		matplot::xlabel(fx3, xlab);
		matplot::ylabel(fx3, "sum of squares");
		matplot::legend(fx3, {});
	}

	matplot::show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Univariate regression\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Let's create some 1D training data
	torch::Tensor x_train = torch::tensor({0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,
                   0.87168699,0.58858043}).to(torch::kDouble);

	torch::Tensor y_train = torch::tensor({-0.25934537,0.18195445,0.651270150,0.13921448,0.09366691,0.30567674,
                    0.372291170,0.20716968,-0.08131792,0.51187806,0.16943738,0.3994327,
                    0.019062570,0.55820410,0.452564960,-0.1183121,0.02957665,-1.24354444,
                    0.248038840,0.26824970}).to(torch::kDouble);

	// Get parameters for the model
	torch::Tensor beta_0, omega_0, beta_1, omega_1;
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	double sigma = 0.2;

	// Define a range of input values
	torch::Tensor x_model = torch::arange(0,1,0.01).to(torch::kDouble);

	// Run the model to get values to plot and plot it.
	torch::Tensor y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	if(plt) plot_univariate_regression(x_model, y_model, x_train, y_train, sigma);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's plot the Gaussian distribution\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	printf("Correct answer = %3.3f, Your answer = %3.3f\n", 0.119, Normal_distribution(torch::tensor({1}), -1, 2.3).data().item<double>());

	//# Let's plot the Gaussian distribution.
	torch::Tensor y_gauss = torch::arange(-5,5,0.1).to(torch::kDouble);
	double mu = 0;
	sigma = 1.0;
	torch::Tensor gauss_prob = Normal_distribution(y_gauss, mu, sigma);
	if(plt) plot_normal_distribution(y_gauss, gauss_prob, "mu=0.0, sigma=1.0");


	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Change to mu=1 and leave sigma=1\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	mu = 1.0;
	sigma = 1.0;
	gauss_prob = Normal_distribution(y_gauss, mu, sigma);
	if(plt) plot_normal_distribution(y_gauss, gauss_prob,"mu=1.0, sigma=1.0");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Leave mu = 0 and change sigma to 2.0\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	mu = 0.0;
	sigma = 2.0;
	gauss_prob = Normal_distribution(y_gauss, mu, sigma);
	if(plt) plot_normal_distribution(y_gauss, gauss_prob, "mu=0.0, sigma=2.0");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Leave mu = 0 and change sigma to 0.5\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	mu = 0.0;
	sigma = 0.5;
	gauss_prob = Normal_distribution(y_gauss, mu, sigma);
	if(plt) plot_normal_distribution(y_gauss, gauss_prob,"mu=0.0, sigma=0.5");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute the likelihood\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Let's test this for a homoscedastic (constant sigma) model
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	// Use our neural network to predict the mean of the Gaussian
	torch::Tensor mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	std::cout << "mu_pred:\n" << mu_pred << '\n';
	// Set the standard deviation to something reasonable
	sigma = 0.2;
	// Compute the likelihood
	torch::Tensor likelihood = compute_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
	// Let's double check we get the right answer before proceeding
	printf("Correct answer = %9.9f, Your answer = %9.9f\n", 0.000010624, likelihood.data().item<double>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute the negative log likelihood\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();

	mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	sigma = 0.2;
	// Compute the negative log likelihood
	torch::Tensor nll = compute_negative_log_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
	// Let's double check we get the right answer before proceeding
	printf("Correct answer = %9.9f, Your answer = %9.9f\n", 11.452419564, nll.data().item<double>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute the sum of squares\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();

	mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);

	//Compute the sum of squares
	torch::Tensor sum_of_squares = compute_sum_of_squares(y_train.squeeze(), mu_pred.squeeze());
	// Let's double check we get the right answer before proceeding
	printf("Correct answer = %9.9f, Your answer = %9.9f\n", 2.020992572, sum_of_squares.data().item<double>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Proceeding the training\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Define a range of values for the parameter
	torch::Tensor beta_1_vals = torch::arange(0,1.0,0.01).to(torch::kDouble);

	// Create some arrays to store the likelihoods, negative log likelihoods and sum of squares
	torch::Tensor likelihoods = torch::zeros_like(beta_1_vals);
	torch::Tensor nlls = torch::zeros_like(beta_1_vals);
	torch::Tensor sum_squares = torch::zeros_like(beta_1_vals);

	// Initialise the parameters
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();
	sigma = 0.2;
	for(auto& count : range(static_cast<int>(beta_1_vals.size(0)), 0)) {
  	    // Set the value for the parameter
		beta_1[0][0] = beta_1_vals[count];
	    // Run the network with new parameters
		torch::Tensor y_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
		mu_pred = y_pred.clone();

  	    // Compute and store the three values
		likelihoods[count] = compute_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
		nlls[count] = compute_negative_log_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
		sum_squares[count] = compute_sum_of_squares(y_train.squeeze(), y_pred.squeeze());
  	    // Draw the model for every 20th parameter setting
		if( count % 20 == 0 ) {
			//Run the model to get values to plot and plot it.
			y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
			std::string title = "beta1=" + std::to_string(beta_1[0][0].data().item<double>());
			if(plt) plot_univariate_regression(x_model, y_model, x_train, y_train, sigma, title);
		}
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the likelihood, negative log likelihood, and least squares\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::vector<double> beta_1_v = tensorTovector(beta_1_vals.squeeze());
	std::vector<double> likelihoods_v = tensorTovector(likelihoods.squeeze());
	std::vector<double> nlls_v = tensorTovector(nlls.squeeze());
	std::vector<double> sum_squares_v = tensorTovector(sum_squares.squeeze());

	std::vector<double> max_beta_x;
	std::vector<double> max_beta_y;
	int max_lk = (torch::argmax(likelihoods.squeeze())).data().item<int>();
	double max_beta = beta_1_v[max_lk];
	for(auto& i : range(static_cast<int>(likelihoods_v.size()), 0)) {
		max_beta_x.push_back(max_beta);
		max_beta_y.push_back(likelihoods_v[i]);
	}
	std::sort(max_beta_y.begin(), max_beta_y.end());

	std::vector<double> min_beta_x;
	std::vector<double> min_beta_y;
	int min_nll = (torch::argmin(nlls.squeeze())).data().item<int>();
	double min_beta = beta_1_v[min_nll];

	for(auto& i : range(static_cast<int>(nlls_v.size()), 0)) {
		min_beta_x.push_back(min_beta);
		min_beta_y.push_back(nlls_v[i]);
	}
	min_beta_x.push_back(min_beta);
	min_beta_y.push_back(0.0);
	std::sort(min_beta_y.begin(), min_beta_y.end());

	std::vector<double> min_beta_x2;
	std::vector<double> min_beta_y2;
	int min_ssq = (torch::argmin(sum_squares.squeeze())).data().item<int>();
	double min_beta2 = beta_1_v[min_ssq];
	for(auto& i : range(static_cast<int>(sum_squares_v.size()), 0)) {
		min_beta_x2.push_back(min_beta2);
		min_beta_y2.push_back(sum_squares_v[i]);
	}
	min_beta_x2.push_back(min_beta2);
	min_beta_y2.push_back(0.0);
	std::sort(min_beta_y2.begin(), min_beta_y2.end());

	if(plt) plot_likelihood_negative_log_likelihood_and_least_squares( beta_1_v, likelihoods_v, max_beta_x, max_beta_y,
			nlls_v, min_beta_x, min_beta_y, sum_squares_v, min_beta_x2, min_beta_y2, "beta_1[0]");

	printf("Maximum likelihood = %3.3f, at beta_1=%3.3f\n", likelihoods_v[max_lk], max_beta);
	printf("Minimum negative log likelihood = %3.3f, at beta_1=%3.3f\n", nlls_v[min_nll], min_beta);
	printf("Least squares = %3.3f, at beta_1=%3.3f\n", sum_squares_v[min_ssq], min_beta2);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the best model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	beta_1[0][0] = beta_1_vals[min_ssq];
	y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	if(plt) plot_univariate_regression(x_model, y_model, x_train, y_train, sigma, "beta1=" + std::to_string((beta_1[0][0]).data().item<double>()));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Define a range of values for the sigma parameter\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor sigma_vals = torch::arange(0.1, 0.5, 0.005).to(torch::kDouble);
	// Create some arrays to store the likelihoods, negative log likelihoods and sum of squares
	likelihoods = torch::zeros_like(sigma_vals);
	nlls = torch::zeros_like(sigma_vals);
	sum_squares = torch::zeros_like(sigma_vals);

	// Initialise the parameters
	std::tie(beta_0, omega_0, beta_1, omega_1) = get_parameters();

	// Might as well set to the best offset
	beta_1[0][0] = 0.27;
	for(auto& count : range( static_cast<int>(sigma_vals.size(0)), 0)) {
		// Set the value for the parameter
	    sigma = sigma_vals[count].data().item<double>();
	    // Run the network with new parameters
	    torch::Tensor y_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1);
	    mu_pred = y_pred.clone();

		// Compute and store the three values
	    likelihoods[count] = compute_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
	    nlls[count] = compute_negative_log_likelihood(y_train.squeeze(), mu_pred.squeeze(), sigma);
	    sum_squares[count] = compute_sum_of_squares(y_train.squeeze(), y_pred.squeeze());

	    // Draw the model for every 20th parameter setting
	    if( count % 20 == 0 ) {
	    	// Run the model to get values to plot and plot it.
	    	y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	    	if(plt) plot_univariate_regression(x_model, y_model, x_train, y_train, sigma, "sigma = " + std::to_string(sigma));
	    }
	}

	std::vector<double> sigma_vals_v = tensorTovector(sigma_vals.squeeze());
	likelihoods_v = tensorTovector(likelihoods.squeeze());
	nlls_v = tensorTovector(nlls.squeeze());
	sum_squares_v = tensorTovector(sum_squares.squeeze());
	std::cout << "sum_squares: \n";
	printVector(sum_squares_v);

	std::vector<double> max_sigma_x;
	std::vector<double> max_sigma_y;
	max_lk = (torch::argmax(likelihoods.squeeze())).data().item<int>();
	double max_sigma = sigma_vals_v[max_lk];
	for(auto& i : range(static_cast<int>(likelihoods_v.size()), 0)) {
		max_sigma_x.push_back(max_sigma);
		max_sigma_y.push_back(likelihoods_v[i]);
	}
	std::sort(max_sigma_y.begin(), max_sigma_y.end());

	std::vector<double> min_sigma_x;
	std::vector<double> min_sigma_y;
	min_nll = (torch::argmin(nlls.squeeze())).data().item<int>();
	double min_sigma = sigma_vals_v[min_nll];
	for(auto& i : range(static_cast<int>(nlls_v.size()), 0)) {
		min_sigma_x.push_back(min_sigma);
		min_sigma_y.push_back(nlls_v[i]);
	}
	min_sigma_x.push_back(min_sigma);
	min_sigma_y.push_back(0.0);
	std::sort(min_sigma_y.begin(), min_sigma_y.end());


	if(plt) plot_likelihood_negative_log_likelihood_and_least_squares(sigma_vals_v, likelihoods_v, max_sigma_x, max_sigma_y,
			nlls_v, min_sigma_x, min_sigma_y);

	// Hopefully, you can see that the maximum of the likelihood fn is at the same position as the minimum negative log likelihood
	// The least squares solution does not depend on sigma, so it's just flat -- no use here.
	// Let's check that:
	printf("Maximum likelihood = %3.3f, at sigma=%3.3f\n", likelihoods_v[max_lk], max_sigma);
	printf("Minimum negative log likelihood = %3.3f, at sigma=%3.3f\n", nlls_v[min_nll], min_sigma);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the best model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	sigma = sigma_vals_v[min_nll];
	y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1);
	if(plt) plot_univariate_regression(x_model, y_model, x_train, y_train, sigma,
								"beta_1=" + std::to_string(0.27) + ", sigma=" + std::to_string(sigma));

	std::cout << "Done!\n";
	return 0;
}
