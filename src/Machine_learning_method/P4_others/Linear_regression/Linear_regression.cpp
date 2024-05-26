/*
 * Linear_regression.cpp
 *
 *  Created on: Apr 24, 2024
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
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

#include <matplot/matplot.h>
using namespace matplot;

class LinearRegression {
public:
	LinearRegression() {
        /*
        :desc lr: Learning Rate
        :desc iteration: Number of iterations over complete data set
        */
        lr = 0.01;
        iterations = 1000;
	}

    torch::Tensor y_pred(torch::Tensor X, torch::Tensor w) {
        /*
        :desc w: weight tensor
        :desc X: input tensor
        */
        return torch::mm(torch::transpose(w, 0, 1), X);
    }

    torch::Tensor  loss(torch::Tensor ypred, torch::Tensor y) {
        // :desc c: cost function - to measure the loss between estimated vs ground truth

    	torch::Tensor l = 1.0 / m * torch::sum(torch::pow(ypred - y, 2));
        return l;
    }

    torch::Tensor  gradient_descent(torch::Tensor w, torch::Tensor X, torch::Tensor y, torch::Tensor ypred) {

        //:desc dCdW: derivative of cost function
        //:desc w_update: change in weight tensor after each iteration

    	torch::Tensor dCdW = 2.0 / m * torch::mm(X, torch::transpose(ypred - y, 0, 1));
    	torch::Tensor w_update = w - lr * dCdW;

        return w_update;
    }

    std::tuple<torch::Tensor, std::vector<double>, std::vector<double>> run(
    																	torch::Tensor X, torch::Tensor y) {
        // :type y: tensor object
        // :type X: tensor object

        auto bias = torch::ones({1, X.size(1)});
        std::vector<double> eph;
        std::vector<double> cst;

        X = torch::cat({bias, X}, 0);
        std::cout << "X: " << X.sizes() << '\n';
        m = X.size(1);
        n = X.size(0);
        auto w = torch::zeros({n, 1});

        for(auto& iteration : range(iterations, 1) ) {
            auto ypred = y_pred(X, w);
            auto cost = loss(ypred, y);

            if( iteration % 100 == 0 ) {
            	std::cout << "Loss at iteration " << iteration << " is " << cost.data().item<double>() << '\n';
            	eph.push_back(iteration*1.0);
            	cst.push_back(cost.data().item<double>());
            }

            w = gradient_descent(w, X, y, ypred);
        }

        return std::make_tuple(w, eph, cst);
    }
private:
        double lr = 0.0;
        int iterations = 0;
        int m =0, n = 0;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

    torch::Tensor X = torch::rand({1, 10});
    std::cout << "X: " << X << '\n';

    torch::Tensor y = 2 * X + 3 + torch::randn({1, 10}) * 0.1;
    std::cout << "y: " << y << '\n';

    auto regression = LinearRegression();
    torch::Tensor w;
	std::vector<double> eph, cst;
    std::tie( w, eph, cst ) = regression.run(X, y);
    std::cout << "w:\n" << w << '\n';

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	plot(ax1, eph, cst)->line_width(2);
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "loss");
	matplot::title(ax1, "Linear Regression");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




