/*
 * Machine_learning_basics.cpp
 *
 *  Created on: Apr 26, 2024
 *      Author: jiamny
 */
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

#include "../../Algorithms/NaiveBayes.h"
#include "../../Utils/csvloader.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

std::pair<torch::Tensor, torch::Tensor> generate_data(torch::Tensor beta, double sig, int64_t n) {
	torch::Tensor u = torch::rand({n, 1});
	u = std::get<0>(torch::sort(u, 0));
	u = u.to(torch::kDouble);
	auto t = torch::arange(0, 4);
	auto m = torch::pow(u, t);
	torch::Tensor y = torch::matmul(m, beta) + sig * torch::randn({n, 1});
	y = y.to(torch::kDouble);
   return std::make_pair(u, y);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 线性回归; Linear regression\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor X = torch::cat(
			{torch::tensor({{-0.5,-0.45,-0.35,-0.35,-0.1,0.0,0.2,0.25,0.3,0.5}}).reshape({-1, 1}),
					torch::ones({10,1}).to(torch::kFloat32)*1.0}, 1);
	torch::Tensor y = torch::tensor({-0.2,0.1,-1.25,-1.2,0.0,0.5,-0.1,0.2,0.5,1.2}).reshape({-1,1});

	//# 用公式求权重
	torch::Tensor w = torch::linalg::inv(X.t().matmul(X)).matmul(X.t()).matmul(y);
	torch::Tensor hat_y = X.matmul(w);
	std::cout << "Weight:\n" << w << '\n';

	torch::Tensor x = torch::linspace(-1, 1, 50);
	hat_y = x * w[0] + w[1];

	auto F = figure(true);
	F->size(600, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	plot(ax, tensorTovector(x.to(torch::kDouble)), tensorTovector(hat_y.to(torch::kDouble)), "r")->line_width(2.5);
	hold(ax, on);
	scatter(ax, tensorTovector(X.index({Slice(),0}).to(torch::kDouble)),
			tensorTovector(y.index({Slice(),0}).to(torch::kDouble)), 10)->marker_face(true);
	xlim(ax, {-1.0, 1.0});
	ylim(ax, {-3, 3});
	ylabel(ax, "y");
	xlabel(ax, "x_1");
	title(ax, "Linear Regression");
	F->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 容量、过拟合、欠拟合; Capacity, over/underﬁtting\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor beta = torch::tensor({{10.0, -140.0, 400.0, -250.0}}).t().to(torch::kDouble);
	std::cout << "beta:\n" << beta.sizes() << '\n';
	int64_t n = 100;
	double sig = 5;

	torch::Tensor U, Y;
	std::tie(U, Y) = generate_data(beta, sig, n);

	vector<double> px = tensorTovector(U.squeeze());
	vector<double> py = tensorTovector(Y.squeeze());
	vector<double> coef = tensorTovector(beta.squeeze());
	vector<double> true_py = tensorTovector(polyf(U, beta).squeeze());

	int p = 2;
	torch::Tensor fbeta = polyfit(U, Y, p);
	printVector(tensorTovector(fbeta));
	vector<double> fpy_2 = tensorTovector(polyf(U, fbeta).squeeze());

	p = 4;
	fbeta = polyfit(U, Y, p);
	printVector(tensorTovector(fbeta));
	vector<double> fpy_4 = tensorTovector(polyf(U, fbeta).squeeze());

	p = 10;
	fbeta = polyfit(U, Y, p);
	printVector(tensorTovector(fbeta));
	vector<double> fpy_10 = tensorTovector(polyf(U, fbeta).squeeze());

	auto F2 = figure(true);
	F2->size(800, 600);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);
	auto fx = F2->nexttile();
	scatter(fx, px, py, 10)->marker_face(true).display_name("data points");
	hold(fx, on);
	plot(fx, px, fpy_2, "b-")->line_width(2.5).display_name("p = 2. underfit");
	plot(fx, px, fpy_4, "k--")->line_width(3.0).display_name("p = 4, correct");
	plot(fx, px, fpy_10, "r-:")->line_width(4.5).display_name("p = 10, overfit");
	xlabel(fx, "x");
	ylabel(fx, "ployfit(x)");
	title(fx, "overfit/underfit");
	legend(fx, {});
	F2->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 朴素贝叶斯; Naive Bayes\n";
	std::cout << "// --------------------------------------------------\n";
	std::ifstream file;
	std::string path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> irisMap;
    irisMap.insert({"Iris-setosa", 0});
    irisMap.insert({"Iris-versicolor", 1});
    irisMap.insert({"Iris-virginica", 2});

    std::cout << "irisMap['Iris-setosa']: " << irisMap["Iris-setosa"] << '\n';
    torch::Tensor IX, Iy;
    std::tie(IX, Iy) = process_data2(file, irisMap, false, false, false);
    IX = IX.index({Slice(0, 100), Slice()});
    Iy = Iy.index({Slice(0, 100), Slice()});
	// change setosa => 0 and versicolor + virginica => 1
	Iy.masked_fill_(Iy > 0, 1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(100, true);
	printVector(tensorTovector(sidx.squeeze().to(torch::kDouble)));

	IX = torch::index_select(IX, 0, sidx.squeeze());
	Iy = torch::index_select(Iy, 0, sidx.squeeze());

	torch::Tensor X_train = IX.index({Slice(0, 70), Slice()});
	torch::Tensor X_test = IX.index({Slice(70, None), Slice()});
	torch::Tensor y_train = Iy.index({Slice(0, 70), Slice()});
	torch::Tensor y_test = Iy.index({Slice(70, None), Slice()});
	std::cout << "Train size = " << X_train.sizes() << "\n";
	std::cout << "Test size = " << X_test.sizes() << "\n";

	NaiveBayes model = NaiveBayes();
	model.fit(X_train, y_train);
	printf("Score = %.3f\n", model.score(X_test, y_test, true));

	std::cout << "Done!\n";
	file.close();
	return 0;
}

