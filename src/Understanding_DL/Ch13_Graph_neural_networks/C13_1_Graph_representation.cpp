/*
 * C12_4_Decoding_strategies.cpp
 *
 *  Created on: Mar 13, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>


#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/UDL_util.h"

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

	torch::Tensor A = torch::tensor(
		{{0,1,0,1,0,0,0,0},
	     {1,0,1,1,1,0,0,0},
	     {0,1,0,0,1,0,0,0},
	     {1,1,0,0,1,0,0,0},
	     {0,1,1,1,0,1,0,1},
	     {0,0,0,0,1,0,1,1},
	     {0,0,0,0,0,1,0,0},
	     {0,0,0,0,1,1,0,0}}).to(torch::kInt32);
	print(A);
	std::cout << '\n';

    std::vector<std::pair<size_t, size_t>> edges = tensorToedges(A);

    auto g = matplot::graph(edges);
    g->show_labels(true);
    g->node_labels({"{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}"});
    g->marker("o");
    g->node_color("red");
    g->marker_size(10);
    g->line_style("-");
    g->line_width(2);

    matplot::show();

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Find how many walks of length three are between nodes 3 and 7\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor A2 = torch::matmul(A, A);
	torch::Tensor A3 = torch::matmul(A2, A);
	print(A3);

	printf("\nNumber of walks of length three between nodes three and seven = %i\n", A3[7][3].data().item<int>());

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Find what the minimum path distance between nodes 0 and 6\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor x = torch::tensor({1,0,0,0,0,0,0,0}).to(torch::kInt32).t();
	torch::Tensor Ax = torch::matmul(A, x);
	torch::Tensor Ai = A.clone();
	if(Ax[6].data().item<int>() > 0 ) {
		printf("The minimum path distance between %i and %i is: %i\n", 0, 6, 1);
	} else {
		int p_dis = 0;
		for(auto& i : range(8, 0)) {
			Ai = torch::matmul(Ai, A);
			Ax = torch::matmul(Ai, x);
			std::cout << "i = " << i << '\n' << Ax << '\n';
			if(Ax[6].data().item<int>() > 0 ) {
				p_dis = i + 1;
				break;
			}
		}
		if( p_dis > 0 ) {
			printf("The minimum path distance between %i and %i is: %i\n", 0, 6, p_dis);
		} else {
			printf("No path between nodes 0 and 6\n");
		}
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// How many paths of length 3 are there between node 0 and every other node\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	Ax = torch::matmul(A3, x);
	print(Ax);
	printf("\nThere are %i paths of length 3 are there between node 0 and every other node.\n", Ax.sum().data().item<int>());

	std::cout << "Done!\n";
	return 0;
}




