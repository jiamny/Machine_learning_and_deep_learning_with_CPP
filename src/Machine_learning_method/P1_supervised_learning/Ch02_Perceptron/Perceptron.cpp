/*
 * SVD.cpp
 *
 *  Created on: May 3, 2024
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
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

std::tuple<torch::Tensor, torch::Tensor> original_form_of_perceptron(torch::Tensor x,
		torch::Tensor y, int64_t eta) {
    /*感知机学习算法的原始形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的w和b
    */
	torch::NoGradGuard no_grad;
    int64_t n_samples = x.size(0); 	 // 样本点数量
    int64_t n_features = x.size(1);  // 特征向量维度数

    torch::Tensor w0 = torch::zeros({n_features}).to(torch::kLong);
    torch::Tensor b0 = torch::tensor({0}).to(torch::kLong);   // 选取初值w0,b0

    while( true ) {  // 不断迭代直至没有误分类点
    	int64_t i = 0;
        for(; i < n_samples; i++) {
        	torch::Tensor xi = x[i];
        	torch::Tensor yi = y[i];
        	torch::Tensor t = yi * (torch::sum(w0 * xi) + b0);

            if( t.data().item<long>() <= 0 ) {
            	torch::Tensor w1 = w0 + eta *yi * xi; //[w0[j] + eta * yi * xi[j] for j in range(n_features)]
            	torch::Tensor b1 = b0 + eta * yi;
                w0 = w1.clone();
                b0 = b1.clone();
                break;
            }
        }
        if(i == n_samples) {
        	return std::make_tuple(w0, b0);
        }
    }
}

torch::Tensor count_gram(torch::Tensor x) {
	/*
     计算Gram矩阵
    :param x: 输入变量
    :return: 输入变量的Gram矩阵
    */
	torch::NoGradGuard no_grad;
    int64_t n_samples = x.size(0); 	// 样本点数量
    int64_t n_features = x.size(1);	// 特征向量维度数
    //gram = [[0] * n_samples for _ in range(n_samples)]  # 初始化Gram矩阵
    torch::Tensor gram = torch::zeros({n_samples, n_samples}).to(torch::kLong);

    // 计算Gram矩阵
    for(int64_t i = 0; i < n_samples; i++) {
        for(int64_t j = i; j < n_samples; j++) {
        	torch::Tensor s = torch::tensor({0}).to(torch::kLong);
        	for(int64_t k = 0; k < n_features; k++) {
        		//gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(n_features))
        		s += x[i][k] * x[j][k];
        	}
        	gram.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({i}),
        															 torch::tensor({j})}, s);
			gram.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({j}),
																	 torch::tensor({i})}, s);
        }
    }
    return gram;
}

std::tuple<torch::Tensor, torch::Tensor> dual_form_perceptron(torch::Tensor x,
		torch::Tensor y, int64_t eta) {
    /*感知机学习算法的对偶形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的a(alpha)和b
    */
    int64_t n_samples = x.size(0);	// 样本点数量
    // 选取初值a0(alpha),b0
    torch::Tensor a0 = torch::zeros({n_samples}).to(torch::kLong);
    torch::Tensor b0 = torch::tensor({0}).to(torch::kLong);
    //a0, b0 = [0] * n_samples, 0  # 选取初值a0(alpha),b0
    torch::Tensor  gram = count_gram(x);  // 计算Gram矩阵

    while(true) {  // 不断迭代直至没有误分类点
    	int64_t i = 0;
        for(; i < n_samples; i++ ) {
            torch::Tensor yi = y[i];
            torch::Tensor val = torch::tensor({0}).to(torch::kLong);
            for(int64_t j = 0; j < n_samples; j++) {
            	torch::Tensor xj = x[j];
            	torch::Tensor yj = y[j];
                val += a0[j] * yj * gram[i][j];
            }

            if( (yi * (val + b0)).data().item<long>() <= 0 ) {
                a0[i] += eta;
                b0 += eta * yi;
                break;
            }
        }
        if(i == n_samples) {
            return std::make_tuple(a0, b0);
        }
    }
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// -----------------------------------------------------------------------\n";
	std::cout << "// 感知机学习算法的原始形式; original form of perceptron\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	torch::Tensor X = torch::tensor({{3, 3}, {4, 3}, {1, 1}}).to(torch::kLong);
	torch::Tensor y = torch::tensor({1, 1, -1}).to(torch::kLong);    // 训练数据集
	int64_t eta = 1;

	torch::Tensor w0, b0;
	std::tie(w0, b0) = original_form_of_perceptron(X, y, eta);
	std::cout << "w0:\n" << w0 << '\n';
	std::cout << "b0:\n" << b0 << '\n';

	std::cout << "// -----------------------------------------------------------------------\n";
	std::cout << "// 感知机学习算法的对偶形式; dual form perceptron\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	eta = 1;
	torch::Tensor a0;
	std::tie(a0, b0) = dual_form_perceptron(X, y, eta);
	std::cout << "a0:\n" << a0 << '\n';
	std::cout << "b0:\n" << b0 << '\n';		// ([2, 0, 5], -3)

	std::cout << "Done!\n";
	return 0;
}
