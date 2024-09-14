/*
 * ShallowNetworkRegions.cpp
 *
 *  Created on: Sep 12, 2024
*      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// Returns value of Binomial Coefficient C(n, k)
long long comb(long long n, long long k) {
	long long res = 1;
    for (int i = n - k + 1; i <= n; ++i)
        res *= i;
    for (int i = 2; i <= k; ++i)
        res /= i;

    return res;
}

long long number_regions(int Di, int D) {
	long long N = 0;
	if( D > Di ) {
		for(int i = 0; i <= Di; i++) {
			//binomial coeﬀicients
			N += comb(D, i);
		}
	} else {
		N = std::pow(2, D);
	}
	return N;
}

// Now let's compute and plot the number of regions as a function of the number of parameters as in figure 3.9b
// First let's write a function that computes the number of parameters as a function of the input dimension and number of hidden units (assuming just one output)

int number_parameters(int D_i, int D) {
  int N = (D_i + 1) * D + D + 1;
  return N;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << number_regions(2, 3) << '\n';
	std::cout << number_regions(10, 50) << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Calculate the number of regions\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	std::cout << number_regions(2, 3) << '\n';
	// Calculate the number of regions for 10D input (Di=10) and 50 hidden units (D=50)
	long long  N = number_regions(10, 50);
	printf("Di=10, D=50, Number of regions = %lld, True value = 13432735556\n", N);

	N = number_regions(10, 8);
	printf("Di=10, D=8, Number of regions = %lld, True value = 256\n", N);

	// Let's do the calculation properly when D<Di (see figure 3.10 from the book)
	int D = 8, Di = 10;
	N = std::pow(2,D);
	// We can equivalently do this by calling number_regions with the D twice
	// Think about why this works
	long long  N2 = number_regions(D,D);
	printf("Di=10, D=8, Number of regions = %lld, Number of regions = %lld, True value = 256\n", N, N2);

	// Now let's plot the graph from figure 3.9a
	torch::Tensor dims = torch::tensor({1,5,10,15});
	torch::Tensor regions = torch::zeros({dims.size(0), 70}).to(torch::kDouble);
	for(auto& c_dim : range(static_cast<int>(dims.size(0)), 0)) {
    	int D_i = dims[c_dim].data().item<int>();
    	printf("Counting regions for %d input dimensions\n", D_i);
    	for(auto& D : range(70, 0)) {
        	regions[c_dim][D] = number_regions(std::min(D_i,D), D)*1.0;
    	}
    	std::cout << "D_i = " << D_i << '\n';
	}
	regions = regions.to(torch::kDouble);

	std::vector<double> xx = iota(0, 70);;
	std::vector<double> y = tensorTovector(regions.index({0, Slice()}).to(torch::kDouble));

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::semilogy(fx, xx, y,"k-")->line_width(2).display_name("D_i=1");
	matplot::hold(fx, true);
	y = tensorTovector(regions.index({1, Slice()}).to(torch::kDouble));

	matplot::semilogy(fx, xx, y,"b-")->line_width(2).display_name("D_i=5");

	y = tensorTovector(regions.index({2, Slice()}).to(torch::kDouble));

	matplot::semilogy(fx, xx, y,"m-")->line_width(2).display_name("D_i=10");

	y = tensorTovector(regions.index({3, Slice()}).to(torch::kDouble));

	matplot::semilogy(fx, xx, y,"c-")->line_width(2).display_name("D_i=15");

	matplot::xlabel(fx, "Number of hidden units, D");
	matplot::ylabel(fx, "Number of regions, N");
	matplot::legend(fx, {})->location(matplot::legend::general_alignment::topleft);
	matplot::show();

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Calculate the number of parameters\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	// Now let's test the code
	int P = number_parameters(10, 8);
	printf("Di=10, D=8, Number of parameters = %d, True value = 97\n", P);
	// Now let's plot the graph from figure 3.9a (takes ~1min)
	dims = torch::tensor({1,5,10,15});
	regions = torch::zeros({dims.size(0), 50}).to(torch::kDouble);
	torch::Tensor params = torch::zeros({dims.size(0), 50}).to(torch::kDouble);

	// We'll compute the five lines separately this time to make it faster
	for(auto&  c_dim : range(static_cast<int>(dims.size(0)), 0)) {
	    int D_i = dims[c_dim].data().item<int>();
	    printf("Counting regions for %d input dimensions\n", D_i);
	    for(auto& c_hidden : range(49, 1)) {
	        // Iterate over different ranges of number hidden variables for different input sizes
	        D = int(c_hidden * 10.0 / D_i);
	        params[c_dim][c_hidden] =  number_parameters(D_i, D)*1.0; //D_i * D + D + D +1
	        regions[c_dim][c_hidden] = number_regions(std::min(D_i,D), D)*1.0;
	    }
	}

	xx = tensorTovector(params.index({0, Slice()}).to(torch::kDouble));
	y = transform(tensorTovector(regions.index({0, Slice()}).to(torch::kDouble)),
										[](auto x) { return log(x); });

	auto F2 = figure(true);
	F2->size(800, 600);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);

	auto ax = F2->nexttile();
	matplot::semilogy(ax, xx, y,"k-")->line_width(2).display_name("D_i=1");
	matplot::hold(true);
	xx = tensorTovector(params.index({1, Slice()}).to(torch::kDouble));
	y = tensorTovector(regions.index({1, Slice()}).to(torch::kDouble));

	matplot::semilogy(ax, xx, y,"b-")->line_width(2).display_name("D_i=5");

	xx = tensorTovector(params.index({2, Slice()}).to(torch::kDouble));
	y = tensorTovector(regions.index({2, Slice()}).to(torch::kDouble));

	matplot::semilogy(ax, xx, y,"m-")->line_width(2).display_name("D_i=10");

	xx = tensorTovector(params.index({3, Slice()}).to(torch::kDouble));
	y = tensorTovector(regions.index({3, Slice()}).to(torch::kDouble));
	//transform(tensorTovector(regions.index({3, Slice()}).to(torch::kDouble)),
	//										[](auto x) { return log(x); });
	matplot::semilogy(ax, xx, y,"c-")->line_width(2).display_name("D_i=15");

	matplot::xlabel(ax, "Number of parameters, D");
	matplot::ylabel(ax, "Number of regions, N");
	matplot::xlim(ax, {0,700});
	matplot::legend(ax, {})->location(matplot::legend::general_alignment::topleft);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




