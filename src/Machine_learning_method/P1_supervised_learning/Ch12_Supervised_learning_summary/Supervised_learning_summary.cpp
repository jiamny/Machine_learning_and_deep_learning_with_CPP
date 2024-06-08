/*
 * Supervised_learning_summary.cpp
 *
 *  Created on: Jun 3, 2024
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
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <float.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 损失函数, loss function\n";
	std::cout << "// --------------------------------------------------\n";

	auto F2 = figure(true);
	F2->size(1000, 800);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);

	//x = np.linspace(start=-1, stop=2, num=1001, dtype=float)
	int n = 1001;
	torch::Tensor x = torch::linspace(-1, 2, n).to(torch::kDouble);

	torch::Tensor logi = torch::log(1 + torch::exp(-x)) / std::log(2);
	torch::Tensor boost = torch::exp(-x);
	torch::Tensor y_01 = x < 0;
	torch::Tensor y_hinge = 1.0 - x;
	y_hinge.masked_fill_((y_hinge < 0), 0); // = 0

	auto fx = F2->nexttile();
	vector<double> fpx = tensorTovector(x.to(torch::kDouble));
	vector<double> y01 = tensorTovector(y_01.to(torch::kDouble));
	vector<double> bst = tensorTovector(boost.to(torch::kDouble));
	vector<double> lgi = tensorTovector(logi.to(torch::kDouble));
	vector<double> yhinge = tensorTovector(y_hinge.to(torch::kDouble));
	printVector(fpx);
	printVector(y01);
	printVector(bst);
	printVector(lgi);
	printVector(yhinge);

	plot(fx, fpx, y01, "g-")->line_width(3).display_name("(0/1损失）0/1 Loss");
	hold(fx, on);
	plot(fx, fpx, yhinge, "b-")->line_width(3.0).display_name("(合页损失）Hinge Loss");
	plot(fx, fpx, bst, "m--")->line_width(2.0).display_name("(指数损失）Adaboost Loss");
	plot(fx, fpx, lgi, "r-")->line_width(2.0).display_name("(逻辑斯谛损失）Logistic Loss");
	xlabel(fx, "函数间隔: yf(x)");
	title(fx, "损失函数");
	legend(fx, {});
	F2->draw();
	show();

	std::cout << "Done!\n";
	return 0;
}
