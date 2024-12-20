/*
 * C6_1_Line_Search.cpp
 *
 *  Created on: Oct 20, 2024
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

// Let's create a simple 1D function
torch::Tensor loss_function(torch::Tensor  phi) {
  return 1- 0.5 * torch::exp(-(phi-0.65)*(phi-0.65)/0.1) - 0.45 *torch::exp(-(phi-0.35)*(phi-0.35)/0.02);
}

void draw_function(torch::Tensor (*loss_function)(torch::Tensor), torch::Tensor a=torch::empty(0),
		torch::Tensor b=torch::empty(0), torch::Tensor c=torch::empty(0), torch::Tensor d=torch::empty(0)) {
    // Plot the function
	torch::Tensor phi_plot = torch::arange(0,1,0.01);
	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	matplot::axes_handle fx = F->nexttile();

	matplot::plot(fx, tensorTovector(phi_plot.to(torch::kDouble)),
			tensorTovector(loss_function(phi_plot).to(torch::kDouble)),"r-")->line_width(3);
	matplot::xlim(fx, {0,1});
	matplot::ylim(fx, {0,1});
	matplot::xlabel(fx, "phi");
	matplot::ylabel(fx, "L[phi]");

    if( a.numel() >= 0 && b.numel() > 0 && c.numel() > 0 && d.numel() > 0) {
    	matplot::hold(fx, true);
    	matplot::plot(fx, std::vector<double> {a.data().item<double>(), a.data().item<double>()}, std::vector<double> {0,1}, "b-")->line_width(2);
    	matplot::plot(fx, std::vector<double> {b.data().item<double>(), b.data().item<double>()}, std::vector<double> {0,1}, "b-")->line_width(2);
    	matplot::plot(fx, std::vector<double> {c.data().item<double>(), c.data().item<double>()}, std::vector<double> {0,1}, "b-")->line_width(2);
    	matplot::plot(fx, std::vector<double> {d.data().item<double>(), d.data().item<double>()}, std::vector<double> {0,1}, "b-")->line_width(2);
    	auto [ta, la] = matplot::textarrow(fx, a.data().item<double>() - 0.15, 0.05, a.data().item<double>(), 0, "a");
    	auto [tb, lb] = matplot::textarrow(fx, b.data().item<double>() - 0.2, 0.1, b.data().item<double>(), 0, "b");
    	auto [tc, lc] = matplot::textarrow(fx, c.data().item<double>() + 0.2, 0.05, c.data().item<double>(), 0, "c");
    	auto [td, ld] = matplot::textarrow(fx, d.data().item<double>() + 0.15, 0.1, d.data().item<double>(), 0, "d");
    	ta->color("red").font_size(14);
    	la->color("k");
    	tb->color("red").font_size(14);
    	lb->color("k");
    	tc->color("red").font_size(14);
    	lc->color("k");
    	td->color("red").font_size(14);
    	ld->color("k");
    }

    matplot::show();
}

double line_search(torch::Tensor (*loss_function)(torch::Tensor), float thresh=.0001, int max_iter = 10, bool draw_flag = false) {

    // Initialize four points along the range we are going to search
    double a = 0;
    double b = 0.33;
    double c = 0.66;
    double d = 1.0;
    int n_iter = 0;

    // While we haven't found the minimum closely enough
    while( std::abs(b-c) > thresh && n_iter < max_iter) {

        // Calculate all four points
        double  lossa = loss_function(torch::tensor({a})).data().item<double>();
        double  lossb = loss_function(torch::tensor({b})).data().item<double>();
        double  lossc = loss_function(torch::tensor({c})).data().item<double>();
        double  lossd = loss_function(torch::tensor({d})).data().item<double>();

        if( draw_flag &&  n_iter % 5 == 0)
          draw_function(loss_function, torch::tensor({a}), torch::tensor({b}), torch::tensor({c}), torch::tensor({d}));

        // Increment iteration counter (just to prevent an infinite loop)
        n_iter = n_iter+1;

        printf("Iter %d, a=%3.3f, b=%3.3f, c=%3.3f, d=%3.3f\n", n_iter, a, b, c, d);

        std::cout << "lossa: "<< lossa << " lossb: " << lossb << " lossc: " << lossc << " lossd: " << lossd << '\n';

        // Rule #1 If the HEIGHT at point A is less than the HEIGHT at points B, C, and D then halve values of B, C, and D
        // i.e. bring them closer to the original point
        // REPLACE THE BLOCK OF CODE BELOW WITH THIS RULE
        if(lossa < lossb && lossa < lossc && lossa < lossd) {
        	b = b/2.0;
        	c = c/2.0;
        	d = d/2.0;
        }

        // Rule #2 If the HEIGHT at point b is less than the HEIGHT at point c then
        //                     point d becomes point c, and
        //                     point b becomes 1/3 between a and new d
        //                     point c becomes 2/3 between a and new d
        // REPLACE THE BLOCK OF CODE BELOW WITH THIS RULE
        if( lossb < lossc) {
        	d = c;
        	double ad = std::max(d, a) - std::min(d, a);
        	b = std::min(d, a) + ad/3.0;
        	c = std::min(d, a) + 2*ad/3.0;
        }

        // Rule #3 If the HEIGHT at point c is less than the HEIGHT at point b then
        //                     point a becomes point b, and
        //                     point b becomes 1/3 between new a and d
        //                     point c becomes 2/3 between new a and d
        // REPLACE THE BLOCK OF CODE BELOW WITH THIS RULE
        if( lossc < lossb) {
        	a = b;
        	double ad = std::max(d, a) - std::min(d, a);
        	b = std::min(d, a) + ad/3.0;
        	c = std::min(d, a) + 2*ad/3.0;
        }
    }

    // FINAL SOLUTION IS AVERAGE OF B and C
    // REPLACE THIS LINE
    double soln = (b + c)/ 2.0;

    return soln;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	bool plt = true;
	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw this function\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	draw_function(loss_function);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Draw line search\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	double soln = line_search(loss_function, .0001, 10, true);
	printf("Soln = %3.5f, loss = %3.5f\n", soln, loss_function(torch::tensor({soln})).data().item<double>());

	std::cout << "Done!\n";
	return 0;
}



