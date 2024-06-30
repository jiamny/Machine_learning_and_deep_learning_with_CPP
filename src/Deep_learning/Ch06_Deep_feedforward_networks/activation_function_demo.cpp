/*
 * activation_function_demo.cpp
 *
 *  Created on: Jun 13, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <matplot/matplot.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/activation.h"


using torch::indexing::Slice;
using torch::indexing::None;

using namespace matplot;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// fake data
	torch::Tensor x = torch::linspace(-5, 5, 200).to(torch::kDouble);  // x data (tensor), shape=(100, 1)

	std::vector<double> x_np = tensorTovector(x);   // numpy array for plotting

	// following are popular activation functions
	_ReLU relu;
	std::vector<double> y_relu = tensorTovector(relu.forward(x).to(torch::kDouble));
	_Sigmoid sigmoid;
	std::vector<double> y_sigmoid = tensorTovector(sigmoid.forward(x).to(torch::kDouble));
	_Tanh tanh;
	std::vector<double> y_tanh = tensorTovector(tanh.forward(x).to(torch::kDouble));
	_SoftPlus softplus;
	std::vector<double> y_softplus = tensorTovector(softplus.forward(x).to(torch::kDouble));

	// plt to visualize these activation function
	auto F = figure(true);
	F->size(1200, 800);
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);

	subplot(2, 2, 0);
	matplot::plot(x_np, y_relu, "r")->line_width(2).display_name("relu");
	matplot::ylim({-1, 5});
	matplot::legend({});

	subplot(2, 2, 1);
	matplot::plot(x_np, y_sigmoid, "r")->line_width(2).display_name("sigmoid");
	matplot::ylim({-0.2, 1.2});
	matplot::legend({});

	subplot(2, 2, 2);
	matplot::plot(x_np, y_tanh, "r")->line_width(2).display_name("tanh");
	matplot::ylim({-1.2, 1.2});
	matplot::legend({});

	subplot(2, 2, 3);
	matplot::plot(x_np, y_softplus, "r")->line_width(2).display_name("softplus");
	matplot::ylim({-0.2, 6});
	matplot::legend({});
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}



