/*
 * C8_4_High_dimensional_spaces.cpp
 *
 *  Created on: Dec 21, 2024
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

float distance_ratio(torch::Tensor x) {
  // -- replace the two lines below to calculate the largest and smallest Euclidean distance between
  // the data points in the columns of x.  DO NOT include the distance between the data point and itself
  // (which is obviously zero)
  float smallest_dist = 100.0;
  float largest_dist = 0.0;
  int sz = x.size(1);

  for(int i = 0; i < sz; i++) {
	  for(int j = 1; j < sz; j++) {
		  if( i < j) {
			  float dis = torch::sqrt(
				  torch::pow(x.index({Slice(),i}) - x.index({Slice(),j}), 2).sum()).data().item<float>();
			  if( dis > largest_dist )
				  largest_dist = dis;

			  if( dis < smallest_dist )
				  smallest_dist = dis;
		  }
	  }
  }

  // Calculate the ratio and return
  float dist_ratio = largest_dist / smallest_dist;
  return dist_ratio;
}

double volume_of_hypersphere(double diameter, long dimensions) {
	double volume = (std::pow(diameter, dimensions) *
			std::pow(M_PI, static_cast<int>(dimensions/2.0)*1.0))/std::tgamma(static_cast<int>(dimensions/2.0) + 1);
	return volume;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// How close are points in high dimensions?\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	int n_data = 1000;
	// Create 1000 data examples (columns) each with 2 dimensions (rows)
	int n_dim = 2;
	torch::Tensor x_2D = torch::normal( 0.0, 1.0, /*size =*/ {n_dim, n_data});

	// Create 1000 data examples (columns) each with 100 dimensions (rows)
	n_dim = 100;
	torch::Tensor x_100D = torch::normal( 0.0, 1.0, /*size =*/ {n_dim, n_data});

	//Create 1000 data examples (columns) each with 1000 dimensions (rows)
	n_dim = 1000;
	torch::Tensor x_1000D = torch::normal( 0.0, 1.0, /*size =*/ {n_dim, n_data});

	printf("Ratio of largest to smallest distance 2D: %3.3f\n", distance_ratio(x_2D));
	printf("Ratio of largest to smallest distance 100D: %3.3f\n", distance_ratio(x_100D));
	printf("Ratio of largest to smallest distance 1000D: %3.3f\n", distance_ratio(x_1000D));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Volume of a hypersphere\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	double diameter = 1.0;
	for(auto& c_dim : range(10, 1)) {
	    printf("Volume of unit diameter hypersphere in %d dimensions is %3.3f\n",
	    		c_dim, volume_of_hypersphere(diameter, c_dim));
	}


	std::cout << "Done!\n";
	return 0;
}





