/*
 * SupervisedLearning.cpp
 *
 *  Created on: Aug 27, 2024
 *      Author: jiamny
 */


#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// Define 1D linear regression model
torch::Tensor f1d(torch::Tensor x, double phi0, double phi1) {
	//the linear regression model (eq 2.4)
	torch::Tensor y = phi0 + phi1 * x;

    return y;
}

// Function to help plot the data
void draw_plot(torch::Tensor x, torch::Tensor y, double phi0, double phi1) {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);
    matplot::scatter(fx, tensorTovector(x), tensorTovector(y), 10)->marker_face(true);
    matplot::xlim(fx, {0, 2.0});
    matplot::ylim(fx, {0, 2.0});
	matplot::xlabel(fx, "Input, x");
	matplot::ylabel(fx, "Output, y");
    // Draw line
    std::vector<double> x_line = linspace(0.0, 2.0, 200);
    std::vector<double> y_line = tensorTovector( f1d(vectorTotensor(x_line), phi0, phi1).to(torch::kDouble));
	matplot::plot(x_line, y_line,"b-")->line_width(2.5);
	matplot::hold(fx, false);
	matplot::show();
}

// Function to calculate the loss
torch::Tensor  compute_loss(torch::Tensor x, torch::Tensor y, double phi0, double phi1) {

	// the loss calculation (equation 2.5)
	torch::Tensor yp = f1d(x, phi0, phi1);
	torch::Tensor loss = torch::sum(torch::pow(yp - y, 2));

	return loss;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  1D linear regression model\n";
	std::cout << "// --------------------------------------------------\n";
	// Create some input / output data
	torch::Tensor x = torch::tensor({0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90}, torch::kDouble);
	torch::Tensor y = torch::tensor({0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 }, torch::kDouble);

	printVector(tensorTovector(x));
	printVector(tensorTovector(y));

	// Set the intercept and slope as in figure 2.2b
	double phi0 = 0.4, phi1 = 0.2;
	// Plot the data and the model
	draw_plot(x, y, phi0, phi1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Compute the loss for our current model\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor loss = compute_loss(x, y, phi0, phi1);
	printf("Your Loss = %3.2f, Ground truth =7.07\n", loss.data().item<double>());

	// Set the intercept and slope as in figure 2.2c
	phi0 = 1.60 ; phi1 =-0.8;
	// Plot the data and the model
	draw_plot(x, y, phi0, phi1);
	loss = compute_loss(x, y, phi0, phi1);
	printf("Your Loss = %3.2f, Ground truth =10.28\n", loss.data().item<double>());

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Find the minimum loss for our current model\n";
	std::cout << "// --------------------------------------------------\n";

	std::vector<double> x_phi0, y_phi1;
	x_phi0.push_back(phi0);
	y_phi1.push_back(phi1);
	double old_loss = loss.data().item<double>();
	double best_phi0 = phi0;
	double best_phi1 = phi1;
	double new_loss  = 100.;
	int count = 0;
	for( double i0 = phi0; i0 > 0.0; i0 -= 0.01 ) {
		double i1 = phi1;
		for(; i1 < 1.0; i1 += 0.01) {
			new_loss = compute_loss(x, y, i0, i1).data().item<double>();
			if( new_loss < old_loss ) {
				best_phi0 = i0;
				best_phi1 = i1;
				old_loss = new_loss;
				count++;
				if( count % 10 == 0) {
					x_phi0.push_back(i0);
					y_phi1.push_back(i1);
				}
			}
		}
		printf("old_loss = %4.6f new_loss = %4.6f best_phi0 = %.3f best_phi1 = %.3f i0 = %.3f i1 = %.3f\n",
				old_loss, new_loss, best_phi0, best_phi1, i0, i1);
	}
	x_phi0.push_back(best_phi0);
	y_phi1.push_back(best_phi1);
	std::cout << "Best phi0 = " << best_phi0 << " Best phi1 = " << best_phi1 << '\n';
	draw_plot(x, y, best_phi0, best_phi1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Visualizing the loss function\n";
	std::cout << "// --------------------------------------------------\n";
	// Make a 2D grid of possible phi0 and phi1 values

	std::vector<std::vector<double>> X, Y, Z;
	for (double i = 0.0; i <= 2.0;  i += 0.02) {
	    std::vector<double> x_row, y_row, z_row;
	    for (double j = -1.0; j <= 1.0; j += 0.02) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            double ls = compute_loss(x, y, i, j).data().item<double>();
	            z_row.push_back(ls);
	    }
	    X.push_back(x_row);
	    Y.push_back(y_row);
	    Z.push_back(z_row);
	}

	std::vector<double> lvls = linspace(-10.0, 10.0, 10);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	matplot::plot(ax, x_phi0, y_phi1, "om-")->line_width(1.5);

	matplot::contour(ax, X, Y, Z)->line_width(3).levels(lvls);;
	matplot::hold(ax, false);
	matplot::xlabel(ax, "Intercept, ϕ0");
	matplot::ylabel(ax, "Slope, ϕ1");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}


