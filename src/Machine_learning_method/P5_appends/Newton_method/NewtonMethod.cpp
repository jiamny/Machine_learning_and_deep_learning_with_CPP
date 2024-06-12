/*
 * NewtonMethod.cpp
 *
 *  Created on: Jun 10, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <functional>
#include <utility> 		// make_pair etc.

#include <matplot/matplot.h>
using namespace matplot;

// ----------------------------------------------
// Newton's Method
const double c = 0.5;

double fn(double x) { return std::cosh(c * x); }

double fn_grad(double x) { return c * std::sinh(c * x); }

double fn_hess(double x) {  // Hessian of the objective function
    return c*c * std::cosh(c * x);
}

std::vector<double> newton( std::function<double(double)> fg, std::function<double(double)> fh, double eta=1) {
    double x = 10.0;
    std::vector<double> results;
    results.push_back(x);
    for( int i = 0; i < 10; i++ ) {
        x -= eta * fg(x) / fh(x);
        results.push_back(x);
    };
    std::cout << "epoch " << 10 << " , x:" <<  x << "\n";
    return results;
}

const double c2 = 0.15 * M_PI;

double fnoncov(double x) { return x * std::cos(c2 * x); }

double fnoncov_grad(double x) {
    return std::cos(c2 * x) - c2 * x * std::sin(c2 * x);
}

double fnoncov_hess(double x) {
    return - 2 * c2 * std::sin(c2 * x) - x * c2*c2 * std::cos(c2 * x);
}

void show_trace(std::vector<double> results, std::function<double(double)> fc) {

	std::vector<double> fresults;
	for(auto& a : results)
		fresults.push_back(fc(a));

	double n = (std::max( std::abs(*min_element(results.begin(), results.end())),
					     std::abs(*max_element(results.begin(), results.end())))) * 1.0;
	std::cout << "n = " << n << "\n";

	std::vector<double> f_line, fx;
	for( double i = (-1*n); i <= n; i += 0.01 ) {
		f_line.push_back(i);
		fx.push_back(fc(i));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, f_line, fx, "b-")->line_width(2);
	matplot::plot(ax1, results, fresults, "m-o")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "f(x)");
	matplot::legend(ax1, {"fx", "fresults"});
	matplot::show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);

	// ----------------------------------------------
	// Adaptive Methods

	std::cout << "Newton's Method - dividing by the Hessian, convex function\n";
	show_trace(newton(&fn_grad, &fn_hess), &fn);

	std::cout << "Newton's Method - dividing by the Hessian, nonconvex function\n";
	show_trace(newton(&fnoncov_grad, &fnoncov_hess), &fnoncov);

	std::cout << "Newton's Method - let us see how this works with a slightly smaller learning rate, say η=0.5.\n";
	show_trace(newton(&fnoncov_grad, &fnoncov_hess, 0.5), &fnoncov);

	std::cout << "Done!\n";
	return 0;
}


