/*
 * C16_1_1D_normalizing_flows.cpp
 *
 *  Created on: Apr 5, 2025
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

// Define the base pdf
torch::Tensor gauss_pdf(torch::Tensor z, float mu, float sigma) {
	torch::Tensor  pr_z = torch::exp( -0.5 * (z-mu) * (z-mu) / (sigma * sigma))/(std::sqrt(2*3.1413) * sigma);
    return pr_z;
}

// Define a function that maps from the base pdf over z to the observed space x
torch::Tensor f(torch::Tensor z) {
	torch::Tensor x1 = 6.0/(1+torch::exp(-(z-0.25)*1.5))-3;
	torch::Tensor x2 = z.clone();
	torch::Tensor p = z * z/9.0;
	torch::Tensor x = (1-p) * x1 + p * x2;
    return x;
}

// Compute gradient of that function using finite differences
torch::Tensor df_dz(torch::Tensor z) {
    return (f(z+0.0001)-f(z-0.0001))/0.0002;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	torch::Tensor z = torch::arange(-3,3,0.01);
	torch::Tensor pr_z = gauss_pdf(z, 0, 1);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::plot(fx, tensorTovector(z.to(torch::kDouble)),  tensorTovector(pr_z.to(torch::kDouble)))->line_width(2);
	matplot::xlim(fx, {-3, 3});
	matplot::xlabel(fx, "z");
	matplot::ylabel(fx, "Pr(z)");
	matplot::show();

	torch::Tensor x = f(z);

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	fx = F->nexttile();
	matplot::plot(fx, tensorTovector(z.to(torch::kDouble)),  tensorTovector(x.to(torch::kDouble)))->line_width(2);
	matplot::xlim(fx, {-3, 3});
	matplot::ylim(fx, {-3, 3});
	matplot::xlabel(fx, "Latent variable, z");
	matplot::ylabel(fx, "Observed variable, x");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





