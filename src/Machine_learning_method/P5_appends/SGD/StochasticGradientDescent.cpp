#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <atomic>
#include <string>
#include <algorithm>
#include <iostream>

#include <matplot/matplot.h>
using namespace matplot;

double f_2d(double x1, double x2) {
    return 0.1 * x1 * x1 + 2 * x2 * x2;
}

void show_trace_2d( std::pair<std::vector<double>, std::vector<double>> rlt, std::string tlt ) {


	auto h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax = h->nexttile();
	matplot::plot(ax, rlt.first, rlt.second, "om-")->line_width(2);
	matplot::hold(ax, true);

	std::vector<std::vector<double>> x, y, z;
	for (double i = -5.5; i <= 1.0;  i += 0.1) {
	    std::vector<double> x_row, y_row, z_row;
	    for(double j = -3.0; j <= 1.0; j += 0.1) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(f_2d(i, j));
	    }
	    x.push_back(x_row);
	    y.push_back(y_row);
	    z.push_back(z_row);
	}

	matplot::contour(ax, x, y, z)->line_width(2);
	matplot::hold(ax, false);
	matplot::xlabel(ax, "x1");
	matplot::ylabel(ax, "x2");
	if( tlt.length() > 1 )
		matplot::title(ax, tlt);
	matplot::show();
}

int t=1;

double constant_lr() {
    return 1.0;
}

double polynomial_lr() {
    // Global variable that is defined outside this function and updated inside
    t += 1;
    return std::pow((1 + 0.1 * t), (-0.5));
}

double exponential_lr() {
    // Global variable that is defined outside this function and updated inside
    t += 1;
    return std::exp(-0.1 * t);
}

using namespace std::chrono;

std::pair<double, double> f_grad(double x1, double x2) {
    return std::make_pair(2 * x1, 4 * x2);
}

std::tuple<double, double, double, double> sgd(double x1, double x2, double s1, double s2,
												double eta, std::string type) {
	double g1, g2;
    std::tie(g1, g2) = f_grad(x1, x2);
    // Simulate noisy gradient
    g1 += torch::normal(0.0, 1, {1}).data().item<double>();
    g2 += torch::normal(0.0, 1, {1}).data().item<double>();
    double eta_t = 0;

    if( type == "constant")
    	eta_t = eta * constant_lr();
    if( type == "exponential")
        eta_t = eta * exponential_lr();
    if( type == "polynomial")
        eta_t = eta * polynomial_lr();

    return std::make_tuple(x1 - eta_t * g1, x2 - eta_t * g2, 0, 0);
}

std::pair<std::vector<double>, std::vector<double>> train_2d(int steps, double eta, std::string type) {
	double x1 = -5, x2 = -2, s1 = 0, s2 = 0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2, s1, s2) = sgd(x1, x2, s1, s2, eta, type);
        x.push_back(x1);
        xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(1000);

	std::cout << torch::normal(0.0, 1, {1}).data().item<double>() << "\n";

	// ------------------------------------------
	// Stochastic Gradient Updates
	// ------------------------------------------
	show_trace_2d( train_2d(50, 0.1, "constant"), "constant" );

	// ------------------------------------------
	// Dynamic Learning Rate
	// ------------------------------------------

	show_trace_2d( train_2d(50, 0.1, "exponential"), "exponential" );

	show_trace_2d( train_2d(50, 0.1, "polynomial"), "polynomial" );

	std::cout << "Done!\n";
	return 0;
}



