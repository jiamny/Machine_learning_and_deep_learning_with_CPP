/*
 * HiddenMarkov_viterbi.cpp
 *
 *  Created on: Jun 2, 2024
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
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

torch::Tensor viterbi_algorithm(torch::Tensor a, torch::Tensor b, torch::Tensor p, torch::Tensor sequence) {
    //维特比算法预测状态序列
    int n_samples = sequence.size(0);
    int n_state = a.size(0);  // 可能的状态数

    // 定义状态矩阵
    torch::Tensor dp = torch::zeros({n_samples, n_state}, torch::kDouble); //[[0.0] * n_state for _ in range(n_samples)];  // 概率最大值
    torch::Tensor last = -1*torch::ones({n_samples, n_state}, torch::kDouble); //[[-1] * n_state for _ in range(n_samples)]  // 上一个结点

    // 处理t=0的情况
    for(auto& i : range(n_state, 0)) {
        dp[0][i] = (p[i] * b[i][sequence[0].data().item<int>()]).data().item<double>();
    }

    // 处理t>0的情况
    for(auto& t : range(n_samples - 1, 1)) {
        for(auto& i : range(n_state, 0)) {
            for(auto& j : range(n_state, 0)) {
                double delta = (dp[t - 1][j] * a[j][i]).data().item<double>();
                if( delta >= dp[t][i].data().item<double>() ) {
                    dp[t][i] = delta;
                    last[t][i] = j;
                }
            }
            dp[t][i] *= (b[i][sequence[t].data().item<int>()]).data().item<double>();
        }
    }

    // 计算最优路径的终点
    int best_end = 0;
    double best_gamma = 0.;
    for(auto& i : range(n_state, 0)){
        if( dp[-1][i].data().item<double>() > best_gamma ) {
            best_end = i;
            best_gamma = dp[-1][i].data().item<double>();
        }
    }

    // 计算最优路径
	torch::Tensor ans = torch::zeros({n_samples}, torch::kInt32); //[0] * (n_samples - 1) + [best_end];
	ans[(n_samples - 1)] = best_end;
    for(int t = (n_samples - 1); t > 0; t--) {
        ans[t - 1] = (last[t][ans[t].data().item<int>()]).data().item<int>();
    }
    return ans;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

    torch::Tensor A = torch::tensor(
    		{{0.5, 0.2, 0.3},
         	 {0.3, 0.5, 0.2},
			 {0.2, 0.3, 0.5}}, torch::kDouble);

    torch::Tensor B = torch::tensor(
    		{{0.5, 0.5},
         	 {0.4, 0.6},
			 {0.7, 0.3}}, torch::kDouble);
	torch::Tensor  p = torch::tensor({0.2, 0.4, 0.4}, torch::kDouble);
	torch::Tensor  seq = torch::tensor({0, 1, 0}, torch::kInt32);

    std::cout << viterbi_algorithm(A, B, p, seq) << '\n'; // [2, 2, 2]

	std::cout << "Done!\n";
	return 0;
}




