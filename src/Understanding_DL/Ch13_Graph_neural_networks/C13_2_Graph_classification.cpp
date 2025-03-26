/*
 * C13_2_Graph_classification.cpp
 *
 *  Created on: Mar 26, 2025
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

// Define the Rectified Linear Unit (ReLU) function
torch::Tensor _ReLU(torch::Tensor preactivation) {
	const std::optional<c10::Scalar> min = {0.0};
	torch::Tensor activation = preactivation.clip(min);
    return activation;
}

torch::Tensor graph_neural_network(torch::Tensor A, torch::Tensor X, torch::Tensor Omega0, torch::Tensor beta0,
								   torch::Tensor Omega1, torch::Tensor beta1, torch::Tensor Omega2, torch::Tensor beta2,
								   torch::Tensor Omega3, torch::Tensor beta3) {
	// Define this network according to equation 13.11 from the book
	int N = A.size(0);
	torch::Tensor t1 = torch::ones({N, 1});
	torch::Tensor I = torch::eye(N);

	torch::Tensor H1 = _ReLU(torch::matmul(beta0, t1.t()) + torch::matmul(torch::matmul(Omega0, X), (A + I)));
	torch::Tensor H2 = _ReLU(torch::matmul(beta1, t1.t()) + torch::matmul(torch::matmul(Omega1, H1), (A + I)));
	torch::Tensor H3 = _ReLU(torch::matmul(beta2, t1.t()) + torch::matmul(torch::matmul(Omega2, H2), (A + I)));
	torch::Tensor f = Sigmoid(beta3 + torch::matmul(torch::matmul(Omega3, H3), t1/N));
	return f;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = false;

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Define the adjacency matrix for this chemical\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	torch::Tensor A = torch::tensor(
		{{0,0,1,0,0,0,0,0,0},
	     {0,0,1,0,0,0,0,0,0},
	     {1,1,0,1,0,1,0,0,0},
	     {0,0,1,0,0,0,0,0,0},
	     {0,0,0,0,0,1,0,0,0},
	     {0,0,0,0,1,0,1,1,0},
	     {0,0,0,0,0,1,0,0,0},
	     {0,0,0,0,0,1,0,0,1},
		 {0,0,0,0,0,0,0,1,0}}).to(torch::kInt32);
	print(A);
	std::cout << '\n';

	if(plt) {
	    std::vector<std::pair<size_t, size_t>> edges = tensorToedges(A);

	    auto g = matplot::graph(edges);
	    g->show_labels(true);
	    g->node_labels({"{0:H}", "{1:H}", "{2:C}", "{3:H}", "{4:H}", "{5:C}", "{6:H}", "{7:O}", "{8:H}"});
	    g->marker("o");
	    g->node_color("red");
	    g->marker_size(10);
	    g->line_style("-");
	    g->line_width(2);

	    matplot::show();
	}

	//
	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Define node matrix\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	/*
	# There will be 9 nodes and 118 possible chemical elements
	# so we'll define a 118x9 matrix.  Each column represents one
	# node and is a one-hot vector (i.e. all zeros, except a single one at the
	# chemical number of the element).
	# Chemical numbers:  Hydrogen-->1, Carbon-->6,  Oxygen-->8
	# Since the indices start at 0, we'll set element 0 to 1 for hydrogen, element 5
	# to one for carbon, and element 7 to one for oxygen
	*/
	torch::Tensor X = torch::zeros({118,9}).to(torch::kInt32);
	std::map<int, std::string> chemNodes = {{0, "H"}, {1, "H"}, {2, "C"}, {3, "H"}, {4, "H"}, {5, "C"}, {6, "H"}, {7, "O"}, {8, "H"}};
	std::map<std::string, int> chemMap = {{"H", 0}, {"C", 5}, {"O", 7}};
	for(auto& r : range(9, 0)) {
		std::string v = chemNodes[r];
		int elm = chemMap[v];
		X[elm][r] = 1;
	}
	print(X.index({Slice(0,15), Slice()}));
	std::cout << '\n';

	// Our network will have K=3 hidden layers, and will use a dimension of D=200.
	int K = 3, D = 200;

	// Let's initialize the parameter matrices randomly with He initialization
	torch::Tensor Omega0 = torch::randn({D, 118}) * 2.0 / D;
	torch::Tensor beta0  = torch::randn({D,1}) * 2.0 / D;
	torch::Tensor Omega1 = torch::randn({D, D}) * 2.0 / D;
	torch::Tensor beta1  = torch::randn({D,1}) * 2.0 / D;
	torch::Tensor Omega2 = torch::randn({D, D}) * 2.0 / D;
	torch::Tensor beta2  = torch::randn({D,1}) * 2.0 / D;
	torch::Tensor Omega3 = torch::randn({1, D});
	torch::Tensor beta3  = torch::randn({1,1});
	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's test this network\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor f = graph_neural_network(A.to(torch::kFloat32), X.to(torch::kFloat32), Omega0, beta0,
											Omega1, beta1, Omega2, beta2, Omega3, beta3);
	printf("Your value is %.5f: %s\n", f[0][0].data().item<float>(), "True value of f: 0.38066");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Let's check that permuting the indices of the graph doesn't change output\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Define a permutation matrix
	torch::Tensor P = torch::tensor({{0,1,0,0,0,0,0,0,0},
	              {0,0,0,0,1,0,0,0,0},
	              {0,0,0,0,0,1,0,0,0},
	              {0,0,0,0,0,0,0,0,1},
	              {1,0,0,0,0,0,0,0,0},
	              {0,0,1,0,0,0,0,0,0},
	              {0,0,0,1,0,0,0,0,0},
	              {0,0,0,0,0,0,0,1,0},
	              {0,0,0,0,0,0,1,0,0}}).to(torch::kInt32);

	// Use this matrix to permute the adjacency matrix A and node matrix X
	torch::Tensor A_permuted = torch::matmul(torch::matmul(P.t(), A), P);
	torch::Tensor X_permuted = torch::matmul(X, P);

	f = graph_neural_network(A_permuted.to(torch::kFloat32), X_permuted.to(torch::kFloat32),
								Omega0, beta0, Omega1, beta1, Omega2, beta2, Omega3, beta3);

	printf("Your value is %.5f: %s\n", f[0][0].data().item<float>(), "True value of f: 0.38066");

	std::cout << "Done!\n";
	return 0;
}




