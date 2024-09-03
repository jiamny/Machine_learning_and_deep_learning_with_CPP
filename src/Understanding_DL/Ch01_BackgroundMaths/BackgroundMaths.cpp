/*
 * BackgroundMaths.cpp
 *
 *  Created on: Aug 26, 2024
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

// Define a linear function with just one input, x
torch::Tensor linear_function_1D(torch::Tensor x, double beta, double omega) {
	// formula for 1D linear equation
	torch::Tensor y = omega * x + beta;

  return y;
}

void draw_1D_function(torch::Tensor x, torch::Tensor y,
		bool annotation = false, std::vector<double> xy={}, std::string txt = "") {
	matplot::plot(tensorTovector(x), tensorTovector(y),"r-")->line_width(2);
	matplot::ylim({0,10});
	matplot::xlim({0,10});
	matplot::xlabel("x");
	matplot::ylabel("y");

	if( annotation ) {
		auto [t, a] = matplot::textarrow(xy[0], xy[1], xy[2], xy[3], txt);
		t->color("blue").font_size(14);
		a->color("blue");
	}

	matplot::show();
}

void draw_2D_function(torch::Tensor x1, torch::Tensor x2, torch::Tensor y, std::vector<double> lvl) {

	std::vector<std::vector<double>> X, Y, Z;
	for(int i = 0; i < x1.size(0); i++) {
		std::vector<double> xx, yy, zz;
		for(int c = 0; c < x1.size(1); c++) {
			xx.push_back((x1[i][c]).data().item<double>());
			yy.push_back((x2[i][c]).data().item<double>());
			zz.push_back((y[i][c]).data().item<double>());
		}
		//printVector(yy);
		X.push_back(xx);
		Y.push_back(yy);
		Z.push_back(zz);
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::contour(fx, X, Y, Z)->line_width(3).line_style("-").levels(lvl);
	matplot::xlabel(fx, "x1");
	matplot::ylabel(fx, "x2");
    matplot::show();
}

// Define a linear function with two inputs, x1 and x2
torch::Tensor linear_function_2D(torch::Tensor x1, torch::Tensor x2, double beta, double omega1, double omega2) {
	// formula for 2D linear equation
	torch::Tensor y = omega1*x1 + omega2*x2 + beta;
	return y;
}

// Define a linear function with three inputs, x1, x2, and x_3
double linear_function_3D(double x1, double x2, double x3,
						  double beta, double omega1, double omega2, double omega3) {
    // formula for a single 3D linear equation
	double y = omega1 * x1 + omega2 * x2 + omega3 * x3 + beta;
    return y;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Tensor x = vectorTotensor(linspace(0.0, 10.0, 1000));
	// Compute y using the function you filled in above
	double beta = 0.0;
	double omega = 1.0;

	torch::Tensor y = linear_function_1D(x, beta, omega);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Plot this function with beta = 0, omega = 1\n";
	std::cout << "// --------------------------------------------------\n";
	draw_1D_function(x, y);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Make a line that crosses y=10 and x=5\n";
	std::cout << "// --------------------------------------------------\n";
	omega = 2.0;
	y = linear_function_1D(x, beta, omega);
	std::vector<double> xy = {7, 8, 5, 10};
	std::string txt = "x=5, y=10";
	draw_1D_function(x, y, true, xy, txt);

	omega = -2.0; beta = 10.0;
	y = linear_function_1D(x, beta, omega);
	draw_1D_function(x, y);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Plot the 2D function\n";
	std::cout << "// --------------------------------------------------\n";
	// Make 2D array of x and y points
	torch::Tensor x1 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);
	torch::Tensor x2 = torch::arange(0.0, 10.0, 0.1).to(torch::kDouble);

	auto vx  = torch::meshgrid( {x1, x2} );
	x2 = vx[0];
	x1 = vx[1];
	// Compute the 2D function for given values of omega1, omega2
	beta = 0.0;
	double  omega1 = 1.0, omega2 = -0.5;
	y  = linear_function_2D(x1, x2, beta, omega1, omega2);

	std::vector<double> levels = linspace(-10.0, 10.0, 20);
	draw_2D_function(x1, x2, y, levels);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Set omega_1 to zero and plot the 2D function\n";
	std::cout << "// --------------------------------------------------\n";

	omega1 = 0.0;
	y  = linear_function_2D(x1, x2, beta, omega1, omega2);
	draw_2D_function(x1, x2, y, levels);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Set omega_2 to zero and plot the 2D function\n";
	std::cout << "// --------------------------------------------------\n";
	omega1 = 1.0;
	omega2 = 0.0;
	y  = linear_function_2D(x1, x2, beta, omega1, omega2);
	draw_2D_function(x1, x2, y, levels);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Compute using the individual equations\n";
	std::cout << "// --------------------------------------------------\n";
	// Define the parameters
	double beta1 = 0.5, beta2 = 0.2,
			omega11 =  -1.0, omega12 = 0.4, omega13 = -0.3,
			omega21 =  0.1, omega22 = 0.1, omega23 = 1.2;

	// Define the inputs
	double xx1 = 4., xx2 = -1., xx3 = 2.;

	// Compute using the individual equations
	double y1 = linear_function_3D(xx1, xx2, xx3, beta1, omega11, omega12, omega13);
	double y2 = linear_function_3D(xx1, xx2, xx3, beta2, omega21, omega22, omega23);
	printf("Individual equations\n");
	printf("y1 = %3.3f\ny2 = %3.3f\n", y1, y2);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Compute with vector/matrix form\n";
	std::cout << "// --------------------------------------------------\n";
	// Define vectors and matrices
	torch::Tensor beta_vec = torch::tensor({{beta1}, {beta2}});
	torch::Tensor omega_mat = torch::tensor({{omega11,omega12,omega13}, {omega21,omega22,omega23}});
	torch::Tensor x_vec = torch::tensor({{xx1}, {xx2}, {xx3}});

	// Compute with vector/matrix form
	torch::Tensor y_vec = beta_vec + torch::matmul(omega_mat, x_vec);
	printf("Matrix/vector form\n");
	printf("y1= %3.3f\ny2 = %3.3f\n", y_vec[0][0].data().item<double>(), y_vec[1][0].data().item<double>());

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Draw the exponential function\n";
	std::cout << "// --------------------------------------------------\n";

	// Define an array of x values from -5 to 5 with increments of 0.01
	x = torch::arange(-5.0,5.0, 0.01);
	y = torch::exp(x) ;

	// Plot this function
	matplot::plot(tensorTovector(x.to(torch::kDouble)), tensorTovector(y.to(torch::kDouble)),"r-")->line_width(3);
	matplot::ylim({0, 100});
	matplot::xlim({-5, 5});
	matplot::xlabel("x");
	matplot::ylabel("exp[x]");
	matplot::show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Draw the logarithm function\n";
	std::cout << "// --------------------------------------------------\n";
	// Define an array of x values from -5 to 5 with increments of 0.01
	x = torch::arange(0.01,5.0, 0.01);
	y = torch::log(x) ;

	// Plot this function
	matplot::plot(tensorTovector(x.to(torch::kDouble)), tensorTovector(y.to(torch::kDouble)),"r-")->line_width(3);
	matplot::ylim({-5,5});
	matplot::xlim({0, 5});
	matplot::xlabel("x");
	matplot::ylabel("$\\log[x]$");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}


