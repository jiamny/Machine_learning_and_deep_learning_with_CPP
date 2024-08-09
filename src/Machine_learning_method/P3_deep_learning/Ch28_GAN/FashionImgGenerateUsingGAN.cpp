/*
 * FashionImgGenerateWithGAN.cpp
 *
 *  Created on: Jul 30, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/opencv_helpfunctions.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/fashion.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	std::cout << "Done!\n";
	return 0;
}




