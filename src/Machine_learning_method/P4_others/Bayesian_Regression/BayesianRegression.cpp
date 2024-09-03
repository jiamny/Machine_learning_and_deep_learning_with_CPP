/*
 * BayesianRegression.cpp
 *
 *  Created on: Aug 29, 2024
 *      Author: jiamny
 */
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor  mean_squared_error(torch::Tensor y_true, torch::Tensor y_pred) {
    // Returns the mean squared error between y_true and y_pred
	torch::Tensor mse = torch::mean(torch::pow(y_true - y_pred, 2));
    return mse;
}


std::vector<torch::Tensor> index_combination(int n_features, int degree) {
	std::vector<torch::Tensor> flat_combinations;
	for(int i = 0; i < n_features; i++ ) {
		std::vector<int> v = {i};
		flat_combinations.push_back(torch::empty(0).to(torch::kInt));
		for(int j = 1; j < (degree+1); j++ ) {
		    std::vector<int> gp(j);
		    combinations_with_replacement(v, j, gp);
		    torch::Tensor idx = torch::from_blob(gp.data(),
		    			{int(gp.size())}, at::TensorOptions(torch::kInt)).clone();
		    flat_combinations.push_back(idx);
		}
	}

    return flat_combinations;
}


torch::Tensor polynomial_features(torch::Tensor X, int degree) {
    /*
    It creates polynomial features from existing set of features. For instance,
    X_1, X_2, X_3 are available features, then polynomial features takes combinations of
    these features to create new feature by doing X_1*X_2, X_1*X_3, X_2*X3.

    For Degree 2:
    combinations output: [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    :param X: Input tensor (For Iris Dataset, (150, 4))
    :param degree: Polynomial degree of 2, i.e we'll have product of two feature vector at max.
    :return: Output tensor (After adding polynomial features, the number of features increases to 15)
    */
    int n_samples  = X.size(0), n_features = X.size(1);


    std::vector<torch::Tensor> combinations = index_combination(n_features, degree);
    int n_output_features = combinations.size();
    torch::Tensor X_new = torch::empty({n_samples, n_output_features});

    int i = 0;
    for(auto& index_combs : combinations) {

    	if( index_combs.numel() < 1) {
    		X_new.index_put_({Slice(), i}, torch::ones(X.size(0)).to(torch::kDouble));
    	} else {
    		X_new.index_put_({Slice(), i}, torch::prod(X.index({Slice(), index_combs}), 1));
    	}
        i++;
    }

    X_new = X_new.to(torch::kDouble);
    return X_new;
}

class BayesianRegression {
public:
	BayesianRegression(int _n_draws, torch::Tensor _mu_0, torch::Tensor _omega_0, int _nu_0,
					   int _sigma_sq_0, int _polynomial_degree=0, double _credible_interval=95) {
        /*
        Bayesian regression model. If poly_degree is specified the features will
        be transformed to with a polynomial basis function, which allows for polynomial
        regression. Assumes Normal prior and likelihood for the weights and scaled inverse
        chi-squared prior and likelihood for the variance of the weights.

        :param n_draws:  The number of simulated draws from the posterior of the parameters.
        :param mu_0:  The mean values of the prior Normal distribution of the parameters.
        :param omega_0: The precision matrix of the prior Normal distribution of the parameters.
        :param nu_0: The degrees of freedom of the prior scaled inverse chi squared distribution.
        :param sigma_sq_0: The scale parameter of the prior scaled inverse chi squared distribution.
        :param polynomial_degree: The polynomial degree that the features should be transformed to. Allows
        for polynomial regression.
        :param credible_interval: The credible interval (ETI in this impl.). 95 => 95% credible interval of the posterior
        of the parameters.
        */
        n_draws = _n_draws;
        polynomial_degree = _polynomial_degree;
        credible_interval = _credible_interval;

        // Prior parameters
        mu_0 = _mu_0;
        omega_0 = _omega_0;
        nu_0 = _nu_0;
        sigma_sq_0 = _sigma_sq_0;
	}

    std::vector<double> scaled_inverse_chi_square(int n, int df, double scale) {
        /*
        Allows for simulation from the scaled inverse chi squared
        distribution. Assumes the variance is distributed according to
        this distribution.
        :param n:
        :param df:
        :param scale:
        :return:
        */
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<double> sigma_sq;
        std::chi_squared_distribution<double> d(static_cast<double>(df));

        for(auto& _ : range(n, 0))
        	sigma_sq.push_back( (df * scale / d(gen)) );

        return sigma_sq;
    }

    void fit(torch::Tensor X, torch::Tensor y) {
        // For polynomial transformation
        if( polynomial_degree > 0 ) {
            X = polynomial_features(X, polynomial_degree);
        }
        std::cout << X.sizes() << '\n';

        int n_samples =  X.size(0), n_features = X.size(1);
        torch::Tensor X_X_T = torch::mm(X.t(), X);
        std::cout << "2: " << X_X_T.sizes() << '\n';

        // Least squares approximate of beta
        torch::Tensor beta_hat = torch::mm(torch::mm(torch::pinverse(X_X_T), X.t()), y);
        std::cout << "3: " << beta_hat.sizes() << '\n';

        // The posterior parameters can be determined analytically since we assume
        // conjugate priors for the likelihoods.
        // Normal prior / likelihood => Normal posterior

        torch::Tensor mu_n = torch::mm(torch::pinverse(X_X_T + omega_0),
        		torch::mm(X_X_T, beta_hat) + torch::mm(omega_0, mu_0.unsqueeze(1)));
        std::cout << "mu_n: " << mu_n.sizes() << '\n';

        torch::Tensor omega_n = X_X_T + omega_0;
        int nu_n = nu_0 + n_samples;
        std::cout << "omega_n: " << omega_n.sizes() << " " << nu_n << '\n';

        // Scaled inverse chi-squared prior / likelihood => Scaled inverse chi-squared posterior
        torch::Tensor sigma_sq_n = (1.0/nu_n) * torch::add(
        		torch::mm(y.t(), y) +
        		torch::mm(torch::mm(mu_0.unsqueeze(1).t(), omega_0), mu_0.unsqueeze(1)) -
				torch::mm(mu_n.t(), torch::mm(omega_n, mu_n)),
				nu_0 * sigma_sq_0);

        // Simulate parameter values for n_draws
        torch::Tensor beta_draws = torch::empty({n_draws, n_features});
        for(auto& i : range(n_draws, 0)) {
            double sigma_sq = scaled_inverse_chi_square(1, nu_n, sigma_sq_n.data().item<double>())[0];

            MultivariateNormalx mnv(mu_n.index({Slice(), 0}).reshape({-1, 1}), sigma_sq * torch::pinverse(omega_n));
            //beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0], cov=sigma_sq * torch.pinverse(omega_n))
            torch::Tensor beta = mnv.rsample();
            //beta_draws[1, :] = torch.tensor(beta,dtype=torch.float)
            beta_draws.index_put_({i, Slice()}, beta.squeeze().to(torch::kDouble));
        }

        // Select the mean of the simulated variables as the ones used to make predictions
        c10::OptionalArrayRef<long int> dim = {0};
        w = torch::mean(beta_draws, dim).to(torch::kDouble);

        // Lower and upper boundary of the credible interval
        double l_eti = 0.50 - credible_interval / 2;
        double u_eti = 0.50 + credible_interval / 2;
        std::vector<torch::Tensor> ts;
        for(auto& i : range(n_features, 0)) {
        	torch::Tensor t = torch::tensor({
        		torch::quantile(beta_draws.index({Slice(), i}), l_eti).data().item<double>(),
        		torch::quantile(beta_draws.index({Slice(), i}), u_eti).data().item<double>()}).reshape({1, -1});
        	ts.push_back(t);
        }
        eti = torch::cat(ts, 0).to(torch::kDouble);

        //eti = torch::tensor([[torch::quantile(beta_draws[:, i], q=l_eti),
        //	torch.quantile(beta_draws[:, i], q=u_eti)] for i in range(n_features)], dtype=torch.double)

    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict(torch::Tensor X, bool _eti=false) {
        if( polynomial_degree > 0 )
            X = polynomial_features(X, polynomial_degree);

        torch::Tensor y_pred = torch::mm(X, w.unsqueeze(1));
        // If the lower and upper boundaries for the 95%
        // equal tail interval should be returned
        if(_eti) {
        	torch::Tensor lower_w = eti.index({Slice(), 0});
			torch::Tensor upper_w = eti.index({Slice(), 1});

			torch::Tensor y_lower_prediction = torch::mm(X, lower_w.unsqueeze(1));
			torch::Tensor y_upper_prediction = torch::mm(X, upper_w.unsqueeze(1));

            return std::make_tuple(y_pred, y_lower_prediction, y_upper_prediction);
        }

        return std::make_tuple(y_pred, torch::empty(0), torch::empty(0));
    }

private:
	int n_draws = 0, polynomial_degree = 0, nu_0 = 0, sigma_sq_0 = 0;
	double credible_interval = 0.;
	torch::Tensor mu_0, omega_0, w, eti ;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Tensor t1 = torch::tensor({  -5.2629,   -3.9980}).unsqueeze(0);
	torch::Tensor t2 = torch::tensor({ -34.4313,  -16.7650}).unsqueeze(0);
	torch::Tensor x = torch::cat({t1, t2}, 0).to(torch::kDouble);
	std::cout << x << '\n';

	std::vector<torch::Tensor> idxs = index_combination(1, 4);
	for(auto& t : idxs)
		std::cout << t << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load data\n";
	std::cout << "// --------------------------------------------------\n";

	std::string file_name = "./data/TempLinkoping2016.txt";
	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<double> X_r;
	std::vector<double> y_r;

	if( fL.is_open() ) {
		 std::getline(fL, line);

		while ( std::getline(fL, line) ) {
			//line = std::regex_replace(line, std::regex("\\\n"), "");
			line = strip(line);
			std::vector<std::string> strs = stringSplit(line, '\t');
			X_r.push_back(std::atof(strip(strs[0]).c_str()));
			y_r.push_back(std::atof(strip(strs[1]).c_str()));
		}
	}
	fL.close();

	torch::Tensor X = torch::from_blob(X_r.data(), {static_cast<int>(X_r.size()), 1}, at::TensorOptions(torch::kDouble)).clone();
	torch::Tensor y = torch::from_blob(y_r.data(), {static_cast<int>(y_r.size()), 1}, at::TensorOptions(torch::kDouble)).clone();
	std::cout << X.sizes() << '\n' << y.sizes() << '\n';


    torch::Tensor x_train, x_test, y_train, y_test;
    std::tie(x_train, x_test, y_train, y_test) = train_test_split(X, y, 0.4, true);
    std::cout << x_train << '\n' << y_train << '\n';

	int n_samples = X.size(0), n_features = X.size(1);
    torch::Tensor mu_0 = torch::zeros(n_features, torch::kDouble);
    std::vector<double> dia;
    for(auto& _ : range(n_features, 0)) {
    	dia.push_back(0.0001);
    }
	torch::Tensor dd = torch::from_blob(dia.data(), {static_cast<int>(dia.size())}, at::TensorOptions(torch::kDouble)).clone();

    torch::Tensor omega_0 = torch::diag(dd).to(torch::kDouble);
    std::cout << "omega_0: " << omega_0.sizes() << '\n';
    int nu_0 = 1, sigma_sq_0 = 100;
    double credible_interval = 0.40;

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Bayesian Regression\n";
	std::cout << "// --------------------------------------------------\n";

    BayesianRegression classifier(2000, mu_0, omega_0, nu_0, sigma_sq_0, 4, credible_interval);

    classifier.fit(x_train, y_train);
    torch::Tensor y_pred = std::get<0>(classifier.predict(x_test));
    torch::Tensor mse = mean_squared_error(y_test, y_pred);

    torch::Tensor y_pred_, y_lower_, y_upper_;
    std::tie(y_pred_, y_lower_, y_upper_) = classifier.predict(X, true);
    std::cout << "Mean Squared Error: " << mse << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Plot the results\n";
	std::cout << "// --------------------------------------------------\n";
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	matplot::scatter(ax, tensorTovector(366 * X.squeeze()),
			tensorTovector(y.squeeze()), 8)->display_name("Origin data");
	matplot::plot(ax, tensorTovector(366 * X.squeeze()), tensorTovector(y_pred_.squeeze()),
			"k-")->line_width(3).display_name("Prediction");
	matplot::plot(ax, tensorTovector(366 * X.squeeze()), tensorTovector(y_lower_.squeeze()),
			"m-.")->line_width(2).display_name("Pred lower");
	matplot::plot(ax, tensorTovector(366 * X.squeeze()), tensorTovector(y_upper_.squeeze()),
			"m--")->line_width(2).display_name("Pred upper");
	matplot::xlim(ax, {0, 366});
	matplot::ylim(ax, {-20, 25});
	matplot::title(ax, "MSE: "+ std::to_string(mse.data().item<double>()) );
	matplot::xlabel(ax, "Day");
	matplot::ylabel(ax, "Temperature in Celcius");
	matplot::legend(ax)->location(matplot::legend::general_alignment::bottomright);
	matplot::hold(ax, false);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}



