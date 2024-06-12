/*
 * MonteCarloIntegration.cpp
 *
 *  Created on: Jun 8, 2024
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
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class MonteCarloIntegration {
public:
	MonteCarloIntegration(std::function<torch::Tensor(torch::Tensor)> _func_f, std::function<torch::Tensor(int)> _func_p) {
        // 所求期望的函数
        func_f = _func_f;
        // 抽样分布的概率密度函数
        func_p = _func_p;
	}

	torch::Tensor solve(int num_samples) {
        /*
        蒙特卡罗积分法
        :param num_samples: 抽样样本数量
        :return: 样本的函数均值
        */
    	torch::Tensor samples = func_p(num_samples);

    	torch::Tensor y = func_f(samples);
        return torch::sum(y) / num_samples;
    }
private:
    std::function<torch::Tensor(torch::Tensor)> func_f;
    std::function<torch::Tensor(int)> func_p;
};

torch::Tensor func_f(torch::Tensor x) {
    //"定义函数f"""
    return torch::pow(x, 2) * std::sqrt(2 * M_PI);
}


torch::Tensor func_p(int n) {
    //"""定义在分布上随机抽样的函数g"""
    return torch::normal(0., 1., {n});
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	//设置样本数量
	int num_samples = 1e6;

	// 使用蒙特卡罗积分法进行求解
	MonteCarloIntegration monte_carlo_integration(func_f, func_p);
	torch::Tensor result = monte_carlo_integration.solve(num_samples);
	std::cout << "抽样样本数量: " << num_samples << '\n';
	std::cout << "近似解: " << result << '\n';

	std::cout << "Done!\n";
	return 0;
}




