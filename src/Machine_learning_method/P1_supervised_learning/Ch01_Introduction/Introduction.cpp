/*
 * SVD.cpp
 *
 *  Created on: May 3, 2024
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
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor real_func(torch::Tensor x) {
    return torch::sin(2*M_PI*x);
}

void fitting(vector<double> fpx, vector<double> y0, vector<double> fpy_real,
			 vector<double> fpy_noise, vector<double> fpy_p, int p = 2) {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	auto fx = F->nexttile();

	scatter(fx, fpx, fpy_noise, 8)->marker_face(true).display_name("data with noise");
	hold(fx, on);
	plot(fx, fpx, y0, "c--")->line_width(1.0).display_name("M=0");
	plot(fx, fpx, fpy_real, "k-.")->line_width(3.0).display_name("real");
	plot(fx, fpx, fpy_p, "r-:")->line_width(4.0).display_name("M=" + std::to_string(p));
	legend(fx, {});
	show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 过拟合、欠拟合; Model selection, over/underﬁtting\n";
	std::cout << "// --------------------------------------------------\n";

	int n = 100;
	torch::Tensor x = torch::linspace(0, 1, n).to(torch::kDouble);
	vector<double> y0 = tensorTovector(torch::zeros({n}).to(torch::kDouble));
	// 加上正态分布噪音的目标函数的值
	torch::Tensor y_ = real_func(x);
	torch::Tensor y_noise = torch::normal(0.0, 0.1, {n}).to(torch::kDouble) + y_;
	vector<double> fpx = tensorTovector(x);
	vector<double> fpy_noise = tensorTovector(y_noise);
	vector<double> fpy_real = tensorTovector(real_func(x));
	x.unsqueeze_(1);
	y_noise.unsqueeze_(1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// M = 1\n";
	std::cout << "// --------------------------------------------------\n";
	int p = 1;
	torch::Tensor fbeta = polyfit(x, y_noise, p);
	vector<double> fpy_1 = tensorTovector(polyf(x, fbeta).squeeze());
	fitting(fpx, y0, fpy_real, fpy_noise, fpy_1, p);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// M = 4\n";
	std::cout << "// --------------------------------------------------\n";
	p = 4;
	fbeta = polyfit(x, y_noise, p);
	vector<double> fpy_4 = tensorTovector(polyf(x, fbeta).squeeze());
	fitting(fpx, y0, fpy_real, fpy_noise, fpy_4, p);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// M = 12\n";
	std::cout << "// --------------------------------------------------\n";
	p = 12;
	fbeta = polyfit(x, y_noise, p);
	vector<double> fpy_12 = tensorTovector(polyf(x, fbeta).squeeze());
	fitting(fpx, y0, fpy_real, fpy_noise, fpy_12, p);

	auto F2 = figure(true);
	F2->size(800, 600);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);
	auto fx = F2->nexttile();
	scatter(fx, fpx, fpy_noise, 8)->marker_face(true).display_name("data points");
	hold(fx, on);
	plot(fx, fpx, fpy_1, "b-")->line_width(2.5).display_name("M = 1. underfit");
	plot(fx, fpx, fpy_4, "k--")->line_width(3.0).display_name("M = 4, suitable");
	plot(fx, fpx, fpy_12, "r-:")->line_width(4.5).display_name("M = 12, overfit");
	xlabel(fx, "x");
	ylabel(fx, "ployfit(x)");
	title(fx, "overfit/underfit");
	legend(fx, {});
	F2->draw();
	show();
	std::cout << "Done!\n";
	return 0;
}



