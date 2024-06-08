/*
 * HiddenMarkov_approximation.cpp
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

torch::Tensor approximation_algorithm(torch::Tensor a, torch::Tensor b, torch::Tensor p, torch::Tensor sequence) {
    // 近似算法预测状态序列
    int n_samples = sequence.size(0);
    int n_state = a.size(0);  // 可能的状态数

    // ---------- 计算：前向概率 ----------
    // 计算初值（定义状态矩阵）
    std::vector<torch::Tensor> alpha;
    torch::Tensor dp = torch::zeros({n_state}, torch::kDouble);
    //dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]
    for(auto& i : range(n_state, 0)) {
    	dp[i] = (p[i] * b[i][sequence[0].data().item<int>()]).data().item<double>();
    }
    alpha.push_back(dp);

    // 递推（状态转移）
    //dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[t]] for i in range(n_state)]
	for(int t = 1; t < n_samples; t++) {
		torch::Tensor d = torch::zeros({n_state}, torch::kDouble);
		for(auto& i : range(n_state, 0)) {
			torch::Tensor s = torch::zeros({n_state}, torch::kDouble);
			for(auto& j : range(n_state, 0))
				s[j] = (a[j][i] * dp[j]).data().item<double>();
			int c = sequence[t].data().item<int>();
			d[i] = torch::sum(s).data().item<double>() * (b[i][c]).data().item<double>();
		}

		alpha.push_back(d.clone());
		dp = d.clone();
	}

    // ---------- 计算：后向概率 ----------
    // 计算初值（定义状态矩阵）
    dp = torch::ones({n_state}, torch::kDouble); //[1] * n_state
    std::vector<torch::Tensor> beta;
    beta.push_back(dp.clone());

    // 递推（状态转移）
    // dp = [sum(a[i][j] * dp[j] * b[j][sequence[t]] for j in range(n_state)) for i in range(n_state)]
	for(int t = (n_samples - 1); t > 0; t--) {
		torch::Tensor d = torch::zeros({n_state}, torch::kDouble);
		for(auto& i : range(n_state, 0)) {
			torch::Tensor s = torch::zeros({n_state}, torch::kDouble);
			for(auto& j : range(n_state, 0))
				s[j] = (a[i][j] * dp[j] * b[j][sequence[t].data().item<int>()]).data().item<double>();
			d[i] = torch::sum(s).data().item<double>();
		}

		beta.push_back(d.clone());
		dp = d.clone();
	}

    std::reverse(beta.begin(), beta.end());

    // 计算最优可能的状态序列
	torch::Tensor ans = torch::zeros({n_samples}, torch::kInt32);
    for(auto& t : range(n_samples, 0)) {
        int min_state = -1;
        double min_gamma = 0.;
        for(auto& i : range(n_state, 0)) {
        	torch::Tensor alp = alpha[t];
        	torch::Tensor bet = beta[t];
            double gamma = (alp[i] * bet[i]).data().item<double>();
            std::cout << i << " " << gamma << '\n';
            if( gamma > min_gamma ) {
                min_state = i;
                //min_gamma = gamma;
            }
        }
        ans[t] = min_state;
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

    std::cout << approximation_algorithm(A, B, p, seq) << '\n'; // [2, 2, 2]
	std::cout << "Done!\n";
	return 0;
}





