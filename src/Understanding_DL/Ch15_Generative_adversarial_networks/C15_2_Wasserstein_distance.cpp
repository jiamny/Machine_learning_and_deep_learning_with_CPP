/*
 * C15_2_Wasserstein_distance.cpp
 *
 *  Created on: Apr 4, 2025
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

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Define two probability distributions\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor p = torch::tensor({5, 3, 2, 1, 8, 7, 5, 9, 2, 1});
	torch::Tensor q = torch::tensor({4, 10,1, 1, 4, 6, 3, 2, 0, 1});
	p = p/torch::sum(p);
	q = q/torch::sum(q);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Draw those distributions\n";
	std::cout << "// --------------------------------------------------\n";
	//#
	auto ax = matplot::subplot(2, 1, 0);
	torch::Tensor x = torch::arange(0, p.size(0), 1);
	matplot::bar(ax, tensorTovector(x.to(torch::kDouble)),  tensorTovector(p.to(torch::kDouble)))->face_color("c");
	matplot::ylim(ax, {0,0.35});
	matplot::ylabel(ax, "p(x=i)");

	auto ax1 = matplot::subplot(2, 1, 1);
	matplot::bar(ax1, tensorTovector(x.to(torch::kDouble)),  tensorTovector(q.to(torch::kDouble)))->face_color("m");
	matplot::ylim(ax1, {0,0.35});
	matplot::ylabel(ax1, "q(x=j)");

	matplot::show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Define the distance matrix from figure 15.8d\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor dist_mat = torch::zeros({10, 10});

	std::vector<std::vector<double>> C;
	for(auto& i : range(10, 0)) {
		std::vector<double> a;
		for(auto& j : range(10, 0)) {
			a.push_back(1.0*std::abs(i - j));
			dist_mat[i][j] = 1.0*std::abs(i - j);
		}
		C.push_back(a);
	}

	auto F = figure(true);
	F->size(600, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	matplot::axes_handle fx = F->nexttile();
	matplot::image(fx, 0, 9, 0, 9, C);
	matplot::title(fx, "Distance |i-j|");
	matplot::xlabel(fx,"q");
	matplot::ylabel(fx, "p");

	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





