/*
 * TSNE.cpp
 *
 *  Created on: Jun 15, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/csvloader.h"
#include "../../../Algorithms/TSNE.h"

#include <matplot/matplot.h>
using namespace matplot;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load data\n";
	std::cout << "// --------------------------------------------------\n";

	std::string file_name = "./data/mnist2500_X.txt";
	std::string line;
	std::ifstream fL(file_name.c_str());

	fL.open(file_name, std::ios_base::in);

	// Exit if file not opened successfully
	if (!fL.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << file_name << std::endl;
		return -1;
	}
	int num_records = std::count(std::istreambuf_iterator<char>(fL), std::istreambuf_iterator<char>(), '\n');
	std::cout << "num_records: " << num_records << '\n';
	int c = 784;
	// set file read from begining
	fL.clear();
	fL.seekg(0, std::ios::beg);

	std::vector<int> trData;

	if( fL.is_open() ) {
		while ( std::getline(fL, line) ) {
			line = strip(line);
			std::vector<std::string> strs = stringSplit(line, ' ');

			for(int i = 0; i < strs.size(); i++)
				trData.push_back(std::atoi(strs[i].c_str()));
		}
	}
	fL.close();

	torch::Tensor X = torch::from_blob(trData.data(), {num_records, c}, c10::TensorOptions(torch::kInt)).clone();
	std::cout << "X: " << X.index({Slice(0, 10), Slice(0, 10)}) << " " << X.sizes() << '\n';

	X = X.to(device);

	file_name = "./data/mnist2500_labels.txt";
	std::ifstream fL2(file_name.c_str());

	std::vector<int> labels;

	if( fL2.is_open() ) {
		while ( std::getline(fL2, line) ) {
			line = strip(line);
			labels.push_back(std::atoi(line.c_str()));
		}
	}
	fL2.close();
	std::cout << "labels: " << labels.size() << '\n';

	X = X.to(torch::kDouble);

	torch::Tensor target = torch::from_blob(labels.data(),
				{static_cast<int>(labels.size())}, c10::TensorOptions(torch::kInt32)).clone();

	TSNE tsne = TSNE(1000);

	torch::NoGradGuard noGrad;

	torch::Tensor Y = tsne.fit_tsne(X, 2, 50, 20.0);
	Y = Y.cpu();

	std::vector<double> xx, yy;
	double xmin = 1000, xmax = -1000, ymin = 1000, ymax = -1000;
	for(auto& i : range(num_records, 0)) {
		double x = (Y[i][0]).data().item<double>(), y = (Y[i][1]).data().item<double>();
		if( x < xmin )
			xmin = x;
		if( x > xmax )
			xmax = x;

		if( y < ymin )
			ymin = y;
		if( y > ymax )
			ymax = y;

		xx.push_back(x);
		yy.push_back(y);
	}
	printVector(xx);
	printVector(yy);

	auto F = figure(true);
	F->size(1200, 1000);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
    auto s = matplot::scatter(fx, xx, yy, 8, labels);
    s->marker_face(true);
    matplot::xlim(fx, {xmin - 5, xmax + 5});
    matplot::ylim(fx, {ymin - 5, ymax + 5});
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}


