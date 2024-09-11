/*
 * ShallowNetworks_I.cpp
 *
 *  Created on: Sep 1, 2024
 *      Author: hhj
 */

#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// axes_handle
void draw_2D_function(matplot::axes_handle ax, std::vector<std::vector<double>> x1_mesh,
		std::vector<std::vector<double>> x2_mesh, std::vector<std::vector<double>> y) {
    //pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
    //ax.set_xlabel('x1');ax.set_ylabel('x2')
    //levels = np.arange(-10,10,1.0)
    //ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
	//matplot::plot(ax, x_phi0, y_phi1, "om-")->line_width(1.5);
	std::vector<double> lvls = linspace(-10.0, 10.0, 20);

	matplot::contour(ax, x1_mesh, x2_mesh, y)->line_width(2).levels(lvls);
	matplot::xlabel(ax, "X1");
	matplot::ylabel(ax, "X2");
	//matplot::show();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "Done!\n";
	return 0;
}


