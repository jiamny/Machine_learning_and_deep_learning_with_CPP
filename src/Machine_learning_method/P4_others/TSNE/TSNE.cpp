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
	torch::Tensor affines = torch::ones({2, 2});
	torch::Tensor mask1 = torch::tensor({{1, 0}, {0, 1}}, torch::kInt32);
	torch::Tensor mask2 = torch::tensor({{0, 1}, {1, 0}}, torch::kInt32);
	affines = (affines + 0.2) * mask1 + (affines + 0.8) * mask2;
	std::optional<c10::Scalar> min = 1e-100;
	std::optional<c10::Scalar> max = torch::max(affines).data().item<double>();
	std::cout << "t:\n" << affines.clip(min, max)  << '\n';


	std::ifstream file;
	std::string path = "./data/diabetes.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		file.close();
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::pair<torch::Tensor, torch::Tensor> datas = process_float_data(file);
	file.close();

	torch::Tensor X = datas.first;
	torch::Tensor y = datas.second;
	printVector(tensorTovector(X[0].to(torch::kDouble)));
	std::cout << "y:\n";
	printVector(tensorTovector(y.to(torch::kDouble)));

	TSNE tsne(2, 5.0, 100, 200);
	torch::Tensor Y = tsne.fit_transform(X);

	std::cout << "Y:\n" << Y << '\n';

	std::vector<double> xx = tensorTovector(Y.index({Slice(), 0}).to(torch::kDouble));
	std::vector<double> yy = tensorTovector(Y.index({Slice(), 1}).to(torch::kDouble));
	torch::Tensor zz, _;
	std::tie(zz, _) = torch::_unique(y);
	printVector(tensorTovector(zz.to(torch::kDouble)));

	auto c = tensorTovector(y.to(torch::kDouble)); //

    auto s = matplot::scatter(xx, yy, 14, c);
    s->marker_face(true);

    matplot::show();

	std::cout << "Done!\n";
	return 0;
}


