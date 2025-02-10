/*
 * C10_2D_Convolution.cpp
 *
 *  Created on: Jan 31, 2025
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

// Perform convolution in PyTorch
torch::Tensor conv_pytorch(torch::Tensor image, torch::Tensor conv_weights, int stride=1, int pad =1) {
    // Do the convolution
	torch::Tensor output_tensor = torch::nn::functional::conv2d(image, conv_weights,
			torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(pad));

    return output_tensor; // (batchSize channelsOut imageHeightOut imageHeightIn)
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Training on CPU.") << '\n';

	int n_batch = 1;
	int image_height = 4;
	int image_width = 6;
	int channels_in = 1;
	int kernel_size = 3;
	int channels_out = 1;


	// Create random input image
	// (batchSize, channelsIn, imageHeightIn, =imageWidthIn)
	torch::Tensor image = torch::randn({n_batch, channels_in, image_height, image_width});
	// Create random convolution kernel weights
	// (channelsOut, channelsIn, kernelHeight, kernelWidth)
	torch::Tensor weights = torch::randn({channels_out, channels_in, kernel_size, kernel_size});

	// Perform convolution using PyTorch
	torch::Tensor  conv_results_pytorch = conv_pytorch(image, weights);
	std::cout << "PyTorch Results:\n" << conv_results_pytorch << '\n';

	n_batch = 1;
	image_height = 12;
	image_width = 10;
	channels_in = 1;
	kernel_size = 3;
	channels_out = 1;
	int stride = 2;

	image = torch::randn({n_batch, channels_in, image_height, image_width});
	weights = torch::randn({channels_out, channels_in, kernel_size, kernel_size});

	// Perform convolution using PyTorch
	conv_results_pytorch = conv_pytorch(image, weights, stride);
	std::cout << "PyTorch Results:\n" << conv_results_pytorch << '\n';

	std::cout << "Done!\n";
	return 0;
}




