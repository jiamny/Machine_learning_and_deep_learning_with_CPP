/*
 * HiddenMarkov_impl.cpp
 *
 *  Created on: May 31, 2024
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> baum_welch_algorithm(
		torch::Tensor sequence, int n_state, int max_iter=100) {
    /*Baum-Welch算法学习隐马尔可夫模型

    :param sequence: 观测序列
    :param n_state: 可能的状态数
    :param max_iter: 最大迭代次数
    :return: A,B,π
    */
    int n_samples = sequence.size(0);  // 样本数
    int n_observation = std::get<0>(at::_unique(sequence)).size(0);  // 可能的观测数

    // ---------- 初始化随机模型参数 ----------
    // 初始化状态转移概率矩阵
    torch::Tensor a = torch::rand({n_state, n_state});
	c10::OptionalArrayRef<long int> da = {1};
	torch::Tensor sa = torch::sum(a, da);
	for(auto& i : range(n_state, 0)) {
		torch::Tensor d = a[i]/sa[i];
		a.index_put_({i, Slice()}, d);
	}

    // 初始化观测概率矩阵
    torch::Tensor b = torch::rand({n_state, n_observation});
	c10::OptionalArrayRef<long int> db = {1};
	torch::Tensor sb = torch::sum(b, db);
	for(auto& i : range(n_observation, 0)) {
		torch::Tensor d = b[i]/sb[i];
		b.index_put_({i, Slice()}, d);
	}

    // 初始化初始状态概率向量
	torch::Tensor p = torch::rand({n_state});
	torch::Tensor sp = torch::sum(p);
	p = p/sp;

	/*
    torch::Tensor a = torch::tensor(
    		{{0.16258558485003846, 0.1416268200835638, 0.3344679194584227, 0.361319675607975},
    		{0.0723822397705396, 0.3490967308514515, 0.29737737980524664, 0.2811436495727622},
			{0.3166514063573354, 0.1899252700760863, 0.38261613647806136, 0.11080718708851703},
			{0.25409688652905804, 0.23204142241230552, 0.33958326125800337, 0.17427842980063302}});

	torch::Tensor b = torch::tensor(
			{{0.7451487708634366, 0.25485122913656333},
			{0.48216103213982153, 0.5178389678601785},
			{0.8652141956782147, 0.13478580432178533},
			{0.7584676590108318, 0.2415323409891682}});
	torch::Tensor p = torch::tensor( {0.45562954897222985, 0.18539988003389493, 0.1939470765342486, 0.1650234944596267});
	*/

    for(auto& _ : range(max_iter, 0)) {
        // ---------- 计算：前向概率 ----------
        // 计算初值（定义状态矩阵）
    	torch::Tensor dp = torch::zeros({n_state}, torch::kDouble);
    	for(auto& i : range(n_state, 0)) {
    		dp[i] = (p[i] * b[i][sequence[0].data().item<int>()]).data().item<double>();
    	}

    	std::vector<torch::Tensor> alpha;
    	alpha.push_back(dp.clone());

        // 递推（状态转移）
    	// dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[t]] for i in range(n_state)]
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

        // ---------- 计算：\gamma_t(i) ----------
        std::vector<torch::Tensor> gamma; // = []
        for(auto& t : range(n_samples, 0)) {
            double sum_ = 0.;
            torch::Tensor lst = torch::zeros({n_state}, torch::kDouble); //[0.0] * n_state
            for(auto& i : range(n_state, 0)) {
            	torch::Tensor alp = alpha[t];
            	torch::Tensor bet = beta[t];
                lst[i] = (alp[i] * bet[i]).data().item<double>();
                sum_ += lst[i].data().item<double>();
            }
            lst.div_(sum_);

            gamma.push_back(lst.clone());
        }

        // ---------- 计算：\xi_t(i,j) ----------
        //lst[i][j] = alpha[t][i] * a[i][j] * b[j][sequence[t + 1]] * beta[t + 1][j]
        std::vector<torch::Tensor> xi; // = []
        for(auto& t : range(n_samples - 1, 0)) {
            //lst = [[0.0] * n_state for _ in range(n_state)]
        	torch::Tensor lst = torch::zeros({n_state, n_state}, torch::kDouble);
            for(auto& i : range(n_state, 0)) {
                for(auto& j : range(n_state, 0)) {
                	torch::Tensor alp = alpha[t];
                	torch::Tensor bet = beta[t + 1];
                    lst[i][j] = (alp[i] * a[i][j] * b[j][sequence[t + 1].data().item<int>()] * bet[j]).data().item<double>();
                }
            }
            lst.div_(torch::sum(lst));
            xi.push_back(lst.clone());
        }

        // ---------- 计算新的状态转移概率矩阵 ----------
        //new_a = [[0.0] * n_state for _ in range(n_state)]
        torch::Tensor new_a = torch::zeros({n_state, n_state}, torch::kDouble);
        for(auto& i : range(n_state, 0)) {
            for(auto& j : range(n_state, 0)) {
                double numerator = 0., denominator = 0.;
                for(auto& t : range(n_samples - 1, 0)) {
                	auto xt = xi[t];
                	auto gt = gamma[t];
                    numerator += (xt[i][j]).data().item<double>();
                    denominator += (gt[i]).data().item<double>();
                }
                new_a[i][j] = numerator / denominator;
            }
        }

        // ---------- 计算新的观测概率矩阵 ----------
        torch::Tensor new_b = torch::zeros({n_state, n_observation}, torch::kDouble);
        for(auto& j : range(n_state, 0)) {
            for(auto& k : range(n_observation, 0)) {
				double numerator = 0., denominator = 0.;
                for(auto& t : range(n_samples, 0)) {
                	auto gt = gamma[t];
                    if( sequence[t].data().item<int>() == k ) {
                        numerator += (gt[j]).data().item<double>();
                    }
                    denominator += (gt[j]).data().item<double>();
                }
                new_b[j][k] = numerator / denominator;
            }
        }

        // ---------- 计算新的初始状态概率向量 ----------
        //new_p = [1 / n_state] * n_state
        torch::Tensor new_p = torch::ones({n_state}, torch::kDouble);
        new_p.div_(n_state);
        auto gt = gamma[0];
        for(auto& i : range(n_state, 0)) {
            new_p[i] = gt[i].data().item<double>();//gamma[0][i]
        }

        a = new_a.clone();
        b = new_b.clone();
        p = new_p.clone();
    }

    return std::make_tuple(a, b, p);
}

torch::Tensor forward_algorithm(torch::Tensor a, torch::Tensor b,
		torch::Tensor p, torch::Tensor sequence) {
    //观测序列概率的前向算法
    int n_state = a.size(0);  // 可能的状态数
    int n_samples = sequence.size(0);

    // 计算初值（定义状态矩阵）
	torch::Tensor dp = torch::zeros({n_state}, torch::kDouble);
	for(auto& i : range(n_state, 0)) {
		dp.index_put_({i}, p[i] * b[i][sequence[0].data().item<int>()]);
	}
	//printVector(tensorTovector(dp));

    // 递推（状态转移）
    for(int k = 1; k < n_samples;  k++ ) {
    	torch::Tensor dt = torch::zeros({n_state}, torch::kDouble);
    	for(auto& i : range(n_state, 0)) {
    		torch::Tensor d = torch::zeros({n_state}, torch::kDouble);
        	for(auto& j : range(n_state, 0)) {
        		d[j] = (a[j][i] * dp[j]).data().item<double>();
        	}
        	int c = sequence[k].data().item<int>();
        	dt[i] = torch::sum(d).data().item<double>() * b[i][c].data().item<double>();
    	}
    	dp = dt.clone();
    }

    return torch::sum(dp);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
/*
	torch::Tensor X = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, torch::kDouble);
	c10::OptionalArrayRef<long int> dim = {1};
	torch::Tensor s = torch::sum(X, dim);
	std::cout << "s: " << s << "\n";

	for(auto& i : range(3, 0)) {
		torch::Tensor d =X[i]/s[i];
		X.index_put_({i, Slice()}, d);
	}

	std::cout << "X: " << X.index({0, Slice()}) << "\n";
	std::cout << "X: " << X[0] << "\n";
*/
	// 根据例10.1的A,B,π生成的观测序列
	torch::Tensor sequence = torch::tensor(
			{0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
            1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
            0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0}, torch::kInt32);

	torch::Tensor A = torch::tensor({{0.0, 1.0, 0.0, 0.0},
									 {0.4, 0.0, 0.6, 0.0},
									 {0.0, 0.4, 0.0, 0.6},
									 {0.0, 0.0, 0.5, 0.5}}, torch::kDouble);
	torch::Tensor B = torch::tensor({{0.5, 0.5},
									 {0.3, 0.7},
									 {0.6, 0.4},
									 {0.8, 0.2}}, torch::kDouble);
	torch::Tensor pi = torch::tensor({0.25, 0.25, 0.25, 0.25}, torch::kDouble);

    std::cout << "生成序列的模型参数下，观测序列出现的概率: " << forward_algorithm(A, B, pi, sequence) << '\n';  // 6.103708248799872e-57

    std::tie(A, B, pi) = baum_welch_algorithm(sequence, 4);
    std::cout << "A:\n" << A << "\n";
    std::cout << "B:\n" << B << "\n";
    std::cout << "pi:\n" << pi << "\n";
    std::cout << "训练结果的模型参数下，观测序列出现的概率:" << forward_algorithm(A, B, pi, sequence) << '\n';

	std::cout << "Done!\n";
	return 0;
}




