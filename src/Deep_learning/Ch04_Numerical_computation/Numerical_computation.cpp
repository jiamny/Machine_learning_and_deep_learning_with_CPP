/*
 * Numerical_computation.cpp
 *
 *  Created on: Apr 25, 2024
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
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

//# 对 log softmax 下溢的处理：
torch::Tensor logsoftmax(torch::Tensor x) {
	torch::Tensor y = x - torch::log(torch::sum(torch::exp(x)));
	return y;
}

//梯度下降法
torch::Tensor matmul_chain(std::vector<torch::Tensor> args) {
	if(args.size() == 0 ) {
		return torch::empty(0);
	}

	torch::Tensor result = args[0];
	for(int i = 1; i < args.size(); i++ ) {
		//std::cout << result.sizes() << " " << args[i].sizes() << '\n';
		result = torch::matmul(result, args[i]);
	}
	return result;
}

torch::Tensor gradient_decent(torch::Tensor x, torch::Tensor A, torch::Tensor b,
								double epsilon, double delta) {
	torch::Tensor dt = matmul_chain(std::vector<torch::Tensor> {A.t(), A, x}) -
			  	  	   matmul_chain(std::vector<torch::Tensor> {A.t(), b});
	int64_t cnt = 1;
	while( torch::norm(dt).data().item<double>() > delta ) {

		x -= epsilon*(dt);
		dt = matmul_chain(std::vector<torch::Tensor> {A.t(), A, x}) -
			 matmul_chain(std::vector<torch::Tensor> {A.t(), b});

		if( cnt % 100000 == 0 )
			std::cout << "Iteration: " << cnt << " norm(dt): " << torch::norm(dt).data().item<double>() << '\n';
		cnt++;
	}
	return x;
}

torch::Tensor newton(torch::Tensor x, torch::Tensor A, torch::Tensor b) {
	torch::Tensor y = torch::linalg::inv(matmul_chain(std::vector<torch::Tensor> {A.t(), A}));
	x = matmul_chain(std::vector<torch::Tensor> {y, A.t(), b});
	return x;
}

// 约束优化，约束解的大小
torch::Tensor Constrained_optimization(torch::Tensor x, torch::Tensor A, torch::Tensor b, double delta) {
	int k = x.size(0);
	torch::Tensor lamb = torch::tensor({0.0}).reshape({1,1});
	int64_t cnt = 1;
	while(torch::abs(torch::matmul(x.t(), x)-1.0).data().item<double>() > delta ) { // delta 设为 5e-2，最优设为 0
		auto y = matmul_chain(std::vector<torch::Tensor> {A.t(), A});
		x = matmul_chain(std::vector<torch::Tensor> {
				torch::linalg::inv(y+2.0*lamb*torch::eye(k)), A.t(), b});
		lamb += torch::matmul(x.t(), x)-1.0;

		if( cnt % 100000 == 0 ) {
			std::cout << "Iteration: " << cnt << " abs(dt): "
			          << torch::abs(torch::matmul(x.t(), x)-1.0).data().item<double>() << '\n';
		}
		cnt++;
	}
	return x;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 上溢和下溢; Overflow and underflow\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor x = torch::tensor({1e7, 1e8, 2e5, 2e7});
	torch::Tensor y = torch::exp(x)/torch::sum(torch::exp(x));
	std::cout << "上溢：\n" << y << '\n';
	x = x - torch::max(x);
	// 减去最大值
	y = torch::exp(x)/torch::sum(torch::exp(x));
	std::cout << "上溢处理：\n" << y << '\n';

	x = torch::tensor({-1e10, -1e9, -2e10, -1e10});
	y = torch::exp(x)/torch::sum(torch::exp(x));
	std::cout << "下溢：\n" << y << '\n';
	x = x - torch::max(x);
	// 减去最大值
	y = torch::exp(x)/torch::sum(torch::exp(x));
	std::cout << "下溢处理：\n" << y << '\n';
	std::cout << "log softmax(x):\n" << torch::log(y) << '\n';
	std::cout << "logsoftmax(x):\n" << logsoftmax(x) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 梯度下降法; Gradient descent\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor x0 = torch::tensor({1.0, 1.0, 1.0}).reshape({3, -1});
	torch::Tensor A = torch::tensor({{1.0, -2.0, 1.0}, {0.0, 2.0, -8.0}, {-4.0, 5.0, 9.0}});
	torch::Tensor b = torch::tensor({0.0, 8.0, -9.0}).reshape({3, -1});
	double epsilon = 0.005, delta = 1e-3;
	// 给定 A，b，真正的解 x 为 [29, 16, 3]

	auto dt = matmul_chain(std::vector<torch::Tensor> {A.t(), A, x0}) -
			  matmul_chain(std::vector<torch::Tensor> {A.t(), b});
	std::cout << "dt：\n" << dt << '\n';
	std::cout << "norm(dt)：\n" << torch::norm(dt).data().item<double>() << '\n';

	torch::Tensor r = gradient_decent(x0, A, b, epsilon, delta);
	std::cout << "gradient_decent solution：\n" << r << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// ⽜顿法; Newton's method\n";
	std::cout << "// --------------------------------------------------\n";
	r = newton(x0, A, b);
	std::cout << "newton solution：\n" << r << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 约束优化; Constrained optimization\n";
	std::cout << "// --------------------------------------------------\n";
	delta = 0.034;
	r = Constrained_optimization(x0, A, b, delta);
	std::cout << "Constrained optimization:\n" << r << '\n';

	std::cout << "Done!\n";
	return 0;
}



