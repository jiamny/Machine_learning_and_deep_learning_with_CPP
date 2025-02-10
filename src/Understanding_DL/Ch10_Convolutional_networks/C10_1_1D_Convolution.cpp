/*
 * C10_1_1D_Convolution.cpp
 *
 *  Created on: Jan 15, 2025
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

void draw_the_signal(std::vector<double> xx, std::vector<double> x, std::vector<double> h = {},
		std::string tlt="", std::string label1 = "", std::string label2 = "") {

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	matplot::axes_handle fx = F->nexttile();
	if(h.size() > 0) {
		matplot::hold(fx, true);
		matplot::plot(fx, xx, x, "k-")->line_width(2).display_name(label1);
		matplot::plot(fx, xx, h, "r-")->line_width(2).display_name(label2);
		matplot::legend(fx, {});
	} else {
		matplot::plot(fx, xx, x, "k-")->line_width(2);
		matplot::ylim(fx, {0, 12});
	}
	matplot::xlim(fx, {0, 11});
	matplot::title(fx, tlt);
	matplot::show();
}

// as in figure 10.2a-c.  Write it yourself, don't call a library routine!
// Don't forget that Python arrays are indexed from zero, not from 1 as in the book figures
torch::Tensor conv_3_1_1_zp(torch::Tensor x, torch::Tensor omega) {
	torch::Tensor x_out = torch::zeros_like(x);
    // write this function
	int n = x.size(0);
	int b = omega.size(0);
	int t = static_cast<int>((b - 1)/2.0);
	for(auto& i : range(n, 0)) {
		torch::Tensor s = torch::zeros({b});
		if( i < t ) {
			s.index_put_({Slice(t, b)}, x.index({Slice(0, b-t)}));
			x_out.index_put_({i}, torch::dot(s, omega));
		} else {
			if( i == (n-1)) {
				s.index_put_({Slice(0, b-t)}, x.index({Slice(i-t, i+1)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			} else {
				s.index_put_({Slice(0, b)}, x.index({Slice(i-t, i+t+1)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			}
		}
	}

    return x_out;
}

// with a convolution kernel size of 3, a stride of 2, and a dilation of 1
// as in figure 10.3a-b.  Write it yourself, don't call a library routine!
torch::Tensor conv_3_2_1_zp(torch::Tensor x, torch::Tensor omega) {
	torch::Tensor x_out = torch::zeros({int(std::ceil(x.size(0)/2.0))});

	int n = x_out.size(0);
	int b = omega.size(0);
	int t = static_cast<int>((b - 1)/2.0);
	int a_stride = 1;
	for(auto& i : range(n, 0)) {
		torch::Tensor s = torch::zeros({b});
		if( i < t ) {
			s.index_put_({Slice(t, b)}, x.index({Slice(0, b-t)}));
			x_out.index_put_({i}, torch::dot(s, omega));
		} else {
			if( i == (n-1)) {
				s.index_put_({Slice(0, b-t)}, x.index({Slice(i-t+a_stride, i+1+a_stride)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			} else {
				s.index_put_({Slice(0, b)}, x.index({Slice(i-t+a_stride, i+t+1+a_stride)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			}
		}
	}

    return x_out;
}

// with a convolution kernel size of 5, a stride of 1, and a dilation of 1
// as in figure 10.3c.  Write it yourself, don't call a library routine!
torch::Tensor conv_5_1_1_zp(torch::Tensor x, torch::Tensor omega) {
	torch::Tensor x_out = torch::zeros_like(x);
    // write this function
	int n = x.size(0);
	int b = omega.size(0);
	int t = static_cast<int>((b - 1)/2.0);

	for(auto& i : range(n, 0)) {
		torch::Tensor s = torch::zeros({b});
		if( i < t ) {
			s.index_put_({Slice(t - i, b)}, x.index({Slice(0, b-t+i)}));
			x_out.index_put_({i}, torch::dot(s, omega));
		} else {
			if( i >= (n-t)) {
				s.index_put_({Slice(0, n - i + t)}, x.index({Slice(i-t, n)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			} else {
				s.index_put_({Slice(0, b)}, x.index({Slice(i-t, i+t+1)}));
				x_out.index_put_({i}, torch::dot(s, omega));
			}
		}
	}

    return x_out;
}

torch::Tensor get_dilated_idx(int p, int t, int d, int n) {
	std::vector<int> idx;
	for(int i = 1; i <= t; i++) {
		if( (p - d*i) >= 0 )
			idx.push_back(p - d*i);
	}

	idx.push_back(p);

	for(int i = 1; i <= t; i++) {
		if( (p + d*i) < n )
			idx.push_back(p + d*i);
	}

	//printVector(idx);
	torch::Tensor sidx = torch::from_blob(idx.data(),
			{int(idx.size())}, at::TensorOptions(torch::kInt)).clone();
	return sidx;
}

// with a convolution kernel size of 3, a stride of 1, and a dilation of 2
// as in figure 10.3d.  Write it yourself, don't call a library routine!
// Don't forget that Python arrays are indexed from zero, not from 1 as in the book figures
torch::Tensor conv_3_1_2_zp(torch::Tensor x, torch::Tensor omega) {
	torch::Tensor x_out = torch::zeros_like(x);
    // write this function
	int n = x.size(0);
	int b = omega.size(0);
	int t = static_cast<int>((b - 1)/2.0);
	int d = 2; // dilation

	for(auto& i : range(n, 0)) {
		torch::Tensor sidx = get_dilated_idx(i, t, d, n);
		//std::cout << sidx << '\n';
		torch::Tensor s = torch::zeros({b});

		if( sidx.size(0) < b ) {
			int m = sidx.size(0);
			if( i < t*d ) {
				s.index_put_({Slice(b - m, b)}, x.index({sidx}));
			} else {
				s.index_put_({Slice(0, m)}, x.index({sidx}));
			}
		} else {
			s = x.index({sidx});
		}
		x_out.index_put_({i}, torch::dot(s, omega));

	}
    return x_out;
}

// Compute matrix in figure 10.4 d
torch::Tensor get_conv_mat_3_1_1_zp(int n_out, torch::Tensor omega) {
	torch::Tensor omega_mat = torch::zeros({n_out,n_out});
	// Fill in this matrix
	for(auto& r :range(n_out, 0)) {
		for(auto& c :range(n_out, 0)) {
			if( r == c) {
				if( (r+1) < n_out) omega_mat.index_put_({r+1, c}, omega[0]);
				omega_mat.index_put_({r, c}, omega[1]);
				if( (c+1) < n_out) omega_mat.index_put_({r, c+1}, omega[2]);
			}
		}
	}

    return omega_mat;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = true;

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw the signal\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor x = torch::tensor({5.2, 5.3, 5.4, 5.1, 10.1, 10.3, 9.9, 10.3, 3.2, 3.4, 3.3, 3.1});
	std::vector<double> xx = {0., 1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
	// Draw the signal
	if(plt) draw_the_signal(xx, tensorTovector(x.to(torch::kDouble)));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Check that you have computed this correctly\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	torch::Tensor omega = torch::tensor({0.33,0.33,0.33});
	torch::Tensor h = conv_3_1_1_zp(x, omega);
	// Check that you have computed this correctly
	printf("Sum of output is: %f3.3, should be 71.1\n", torch::sum(h).data().item<float>());
	if(plt) {
		draw_the_signal(xx, tensorTovector(x.to(torch::kDouble)),
			tensorTovector(h.to(torch::kDouble)), "h", "before", "after");
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Change omega\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	omega = torch::tensor({-0.5,0.0,0.5});
	torch::Tensor h2 = conv_3_1_1_zp(x, omega);
	if(plt) {
		draw_the_signal(xx, tensorTovector(x.to(torch::kDouble)),
				tensorTovector(h2.to(torch::kDouble)), "h2", "before", "after");
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Convolution kernel size of 3, a stride of 2, and a dilation of 1\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	omega = torch::tensor({0.33,0.33,0.33});
	torch::Tensor h3 = conv_3_2_1_zp(x, omega);
	std::cout << "h:\n" << h << "h3:\n" << h3 << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// convolution kernel size of 5, a stride of 1, and a dilation of 1\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	omega = torch::tensor({0.2, 0.2, 0.2, 0.2, 0.2});

	torch::Tensor h4 = conv_5_1_1_zp(x, omega);
	// Check that you have computed this correctly
	printf("Sum of output is: %f3.3, should be 69.6\n", torch::sum(h4).data().item<float>());

	if(plt) {
		draw_the_signal(xx, tensorTovector(x.to(torch::kDouble)),
				tensorTovector(h4.to(torch::kDouble)), "h4", "before", "after");
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// convolution kernel size of 3, a stride of 1, and a dilation of 2\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	omega = torch::tensor({0.33,0.33,0.33});
	torch::Tensor h5 = conv_3_1_2_zp(x, omega);

	// Check that you have computed this correctly
	printf("Sum of output is: %f3.3, should be 66.3\n", torch::sum(h5).data().item<float>());

	if(plt) {
		draw_the_signal(xx, tensorTovector(x.to(torch::kDouble)),
				tensorTovector(h5.to(torch::kDouble)), "h5", "before", "after");
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute matrix in figure 10.4 d\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	omega = torch::tensor({-1.0,0.5,-0.2});

	torch::Tensor h6 = conv_3_1_1_zp(x, omega);
	std::cout << "h6:\n";
	printVector(tensorTovector(h6.to(torch::kDouble)));

	// If you have done this right, you should get the same answer
	torch::Tensor omega_mat = get_conv_mat_3_1_1_zp(x.size(0), omega);
	torch::Tensor h7 = torch::matmul(omega_mat, x);
	std::cout <<"h7:\n";
	printVector(tensorTovector(h7.to(torch::kDouble)));

	std::cout << "Done!\n";
	return 0;
}



