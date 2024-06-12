/*
 * MCMC.cpp
 *
 *  Created on: Jun 9, 2024
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
#include <set>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class UniformDistribution {
    /*均匀分布
    :param a: 左侧边界
    :param b: 右侧边界
    */
public:
	UniformDistribution(double _a, double _b) {
        a = _a;
        b = _b;
	}

	double pdf(double x) {
        if( a < x < b )
            return 1. / (b - a);
        else
            return 0.;
    }

	double cdf(double x) {
        if( x < a)
            return 0.;
        else if( a <= x && x < b)
            return (x - a)*1.0 / (b - a);
        else
            return 1.;
    }
private:
	double a = 0, b = 0;
};

class GaussianDistribution {
    /*高斯分布（正态分布）
    :param u: 均值
    :param s: 标准差
    */
public:
	GaussianDistribution(double _u, double _s) {
        u = _u;
        s = _s;
	}

	double pdf(double x) {
        return std::pow(M_PI, -1 * (std::pow(x - u, 2)) / 2 * std::pow(s, 2)) / (std::sqrt(2 * M_PI * std::pow(s, 2)));
    }
private:
	double u, s;
};

std::list<double> direct_sampling_method(UniformDistribution distribution, int n_samples, double a=-1e5, double b=1e5, double tol=1e-6, int random_state=0) {
    /* 直接抽样法抽取样本
    :param distribution: 定义分布函数的概率分布
    :param n_samples: 样本数
    :param a: 定义域左侧边界
    :param b: 定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    */

	torch::manual_seed(random_state);

    std::list<double> samples;
    for(auto& _ : range(n_samples, 0)) {
        double y = torch::rand({1}).data().item<double>();

        // 二分查找解方程：F(x) = y
        double l = a, r = b;
        while( (r - l) > tol) {
            double m = (l + r) / 2.0;

            if( distribution.cdf(m) > y )
                r = m;
            else
                l = m;
        }

        samples.push_back((l + r) / 2.0);
    }

    return samples;
}


std::list<double> accept_reject_sampling_method(GaussianDistribution d1, UniformDistribution d2, double c,
		int n_samples, double a=-1e5, double b=1e5, double tol=1e-6, int random_state=0) {
    /*接受-拒绝抽样法
    :param d1: 目标概率分布
    :param d2: 建议概率分布
    :param c: 参数c
    :param n_samples: 样本数
    :param a: 建议概率分布定义域左侧边界
    :param b: 建议概率分布定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
	*/

	torch::manual_seed(random_state);

	std::list<double> samples;

	// 直接抽样法得到建议分布的样本
	std::list<double>  waiting = direct_sampling_method(d2, n_samples * 2, a, b, tol, random_state);
	//int cnt = 0;
    while( samples.size() < n_samples ) {
        if( waiting.empty() )
            waiting = direct_sampling_method(d2, (n_samples - samples.size()) * 2, a, b, tol, random_state);

        double x = waiting.back();
        waiting.pop_back();
        double u = torch::rand({1}).data().item<double>();
        //std::cout << "x: " << x << " d1.pdf(x): " << d1.pdf(x) << " (c * d2.pdf(x))): " << (c * d2.pdf(x)) << '\n';
        if( u <= (d1.pdf(x) / (c * d2.pdf(x))))
            samples.push_back(x);

        //cnt += 1;
        //if(cnt > 10000)
        //	break;
    }

    return samples;
}

torch::Tensor get_stationary_distribution(torch::Tensor P, double tol=1e-8, int max_iter=1000) {
    /*迭代法求离散有限状态马尔可夫链的某个平稳分布

    根据平稳分布的定义求平稳分布。如果有无穷多个平稳分布，则返回其中任意一个。如果不存在平稳分布，则无法收敛。

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 平稳分布
    */
    int n_components = P.size(0);

    // 初始状态分布：均匀分布
    torch::Tensor pi0 = torch::ones({n_components, 1}, torch::kDouble ) * (1.0 / n_components);

    // 迭代寻找平稳状态
    for(auto& _ : range(max_iter, 0)) {
    	torch::Tensor pi1 = torch::mm(P, pi0);

        // 判断迭代更新量是否小于容差
        if( torch::sum(torch::abs(pi0 - pi1)).data().item<double>() < tol )
            break;

        pi0 = pi1;
    }
    return pi0;
}


bool is_reversible(torch::Tensor P, double tol=1e-4, int max_iter=1000) {
    /*计算有限状态马尔可夫链是否可逆

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 可逆 = True ; 不可逆 = False
    */
    int n_components = P.size(0);
    torch::Tensor D = get_stationary_distribution(P, std::pow(tol, 2), max_iter);  // 计算平稳分布
    for(auto& i : range(n_components, 0)) {
        for(auto& j : range(n_components, 0)) {
        	double d = (P[i][j] * D[j] - P[j][i] * D[i]).data().item<double>();
            if( d <= (-1 * tol) || d >= tol )
                return false;
        }
    }
    return true;
}

bool is_reducible(torch::Tensor P) {
    /*计算马尔可夫链是否可约

    :param P: 转移概率矩阵
    :return: 可约 = True ; 不可约 = False
    */
	 int n_components = P.size(0);

    // 遍历所有状态k，检查从状态k出发能否到达任意状态
    for(auto& k : range(n_components, 0)) {
        std::unordered_set<std::vector<bool>> visited; // = set()  // 当前已遍历过的状态

        bool find = false;  // 当前是否已找到可到达任意位置的时刻
        std::vector<bool> stat0;
        //stat0 = (False,) * k + (True,) + (False,) * (n_components - k - 1)  // 时刻0可达到的位置
        if( k > 0 ) {
        	for(auto& _ : range(k, 0))
        		stat0.push_back(false);
        }
        stat0.push_back(true);
        if( (n_components - k - 1) > 0 ) {
        	for(auto& _ : range((n_components - k - 1), 0))
        		stat0.push_back(false);
        }

        while( visited.find(stat0) == visited.end() ) { //stat0 not in visited ) {

        	visited.insert(stat0);

        	std::vector<bool> stat1; // = [False] * n_components
        	for(auto& _ : range(n_components, 0))
        		stat1.push_back(false);

            for(auto& j : range(n_components, 0)) {
                if( stat0[j] ) {
                    for(auto& i : range(n_components, 0)) {
                        if( P[i][j].data().item<double>() > 0 )
                            stat1[i] = true;
                    }
                }
            }

            // 如果已经到达之前已检查可到达任意状态的状态，则不再继续寻找
            for(auto& i : range(k, 0)) {
                if( stat1[i] ) {
                    find = true;
                    break;
                }
            }

            // 如果当前时刻可到达任意位置，则不再寻找
            bool all_true = true;
            for(int n = 0; n < stat1.size(); n++) {
            	if( ! stat1[n] ) {
            		all_true = false;
            		break;
            	}
            }
            if( all_true ) {
                find = true;
                break;
            }

            stat0 = stat1;
        }

        if(! find )
            return true;
    }

    return false;
}

bool is_periodic(torch::Tensor P) {
    /*计算马尔可夫链是否有周期性

    :param P: 转移概率矩阵
    :return: 有周期性 = True ; 无周期性 = False
    */
	int n_components = P.size(0);

    // 0步转移概率矩阵
	torch::Tensor P0 = P.clone();
	torch::Tensor hash_P = P0.flatten();
	std::string hash_P_str = join(tensorTovector(hash_P.to(torch::kDouble)), "_");

    // 每一个状态上一次返回状态的时刻的最大公因数
	torch::Tensor gcd = torch::zeros({n_components}, torch::kInt32);

    std::map<std::string, int> visited; // = Counter()  // 已遍历过的t步转移概率矩阵
    torch::Tensor t =  torch::tensor({1}, torch::kInt32);  // 当前时刻t

    // 不断遍历时刻t，直至满足如下条件：当前t步转移矩阵之前已出现过2次（至少2次完整的循环）
    while( visited.find(hash_P_str) == visited.end() || visited[hash_P_str] < 2 ) {
        visited[hash_P_str] += 1;

        // 记录当前返回状态的状态
        for(auto& i : range(n_components, 0)) {
            if( P0[i][i].data().item<double>() > 0 ) {
                if( gcd[i].data().item<int>() == 0 )	// 状态i出发时，还从未返回过状态i
                    gcd[i] = t.data().item<int>();
                else  // 计算最大公约数
                    gcd[i] = torch::gcd(gcd[i], t).data().item<int>();
            }
        }

        // 检查当前时刻是否还有未返回(gcd[i]=0)或返回状态的所有时间长的最大公因数大于1(gcd[i]>1)的状态
        for(auto& i : range(n_components, 0)) {
            if( gcd[i].data().item<int>() == 0 || gcd[i].data().item<int>() > 1 )
                break;
			else
				return false;
        }
        // 计算(t+1)步转移概率矩阵
        torch::Tensor P1 = torch::mm(P0, P);

        P0 = P1;
        hash_P = P0.flatten();
        hash_P_str = join(tensorTovector(hash_P.to(torch::kDouble)), "_");
        t.add_(1);
    }
    return true;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Direct sampling method\n";
	std::cout << "// --------------------------------------------------\n";
    UniformDistribution distribution(-3, 3);

    std::list<double> samples = direct_sampling_method(distribution, 10, -3, 3);
    std::vector<double> DS(samples.begin(), samples.end());
    printVector(DS);  // [-0.0224601, 1.60933, -2.46914, -2.20782, -1.15546, 0.804472, -0.0594395, 2.37867, -0.266232, 0.793838]

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Accept reject sampling method\n";
	std::cout << "// --------------------------------------------------\n";

    GaussianDistribution d1(0, 1);
    UniformDistribution  d2(-3, 3);

    double c = (1.0 / std::sqrt(2 * M_PI)) / (1.0 / 6);  // 计算c的最小值
    samples = accept_reject_sampling_method(d1, d2, c, 10, -3, 3);
    // [0.111131, -0.589696, -0.906639, 0.793838, -0.266232, -0.0594395, 0.804472, -1.15546, -0.0224601, 1.60933]
    std::vector<double> ARS(samples.begin(), samples.end());
    printVector(ARS);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Get stationary distribution\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor P = torch::tensor(
				 {{0.5, 0.5, 0.25},
                  {0.25, 0., 0.25},
                  {0.25, 0.5, 0.5}}, torch::kDouble);

    std::cout << get_stationary_distribution(P) << '\n';  // [0.4 0.2 0.4]

    P = torch::tensor(
			 	 {{1., 1. / 3., 0.},
                  {0., 1. / 3., 0.},
                  {0., 1. / 3., 1.}}, torch::kDouble);

    std::cout << get_stationary_distribution(P) << '\n';  // [0.5  0.  0.5]

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Is MCMC reversible\n";
	std::cout << "// --------------------------------------------------\n";

    P = torch::tensor(
		 	 	 {{0.5, 0.5, 0.25},
                  {0.25, 0., 0.25},
                  {0.25, 0.5, 0.5}}, torch::kDouble);

    std::cout << is_reversible(P) << '\n';  // True

    P = torch::tensor(
	 	 	 	 {{0.25, 0.5, 0.25},
                  {0.25, 0., 0.5},
                  {0.5, 0.5, 0.25}}, torch::kDouble);

    std::cout << is_reversible(P) << '\n';  // False

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Is MCMC reducible\n";
	std::cout << "// --------------------------------------------------\n";
    P =  torch::tensor(
	 	 	 	 {{0.5, 0.5, 0.25},
                  {0.25, 0., 0.25},
                  {0.25, 0.5, 0.5}}, torch::kDouble);

	std::cout << is_reducible(P) << '\n';   // False

    P = torch::tensor(
	 	 	 	 {{0., 0.5, 0.},
                  {1., 0., 0.},
                  {0., 0.5, 1.}}, torch::kDouble);

	std::cout << is_reducible(P) << '\n';   // True

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Is MCMC periodic\n";
	std::cout << "// --------------------------------------------------\n";
    P = torch::tensor(
	 	 	 	 {{0.5, 0.5, 0.25},
                  {0.25, 0., 0.25},
                  {0.25, 0.5, 0.5}}, torch::kDouble);

    printVector(tensorTovector(P.flatten()));

    torch::Tensor hash_P = P.flatten();
    std::string   hash_P_str = join(tensorTovector(hash_P.to(torch::kDouble)), "_");
    std::cout << "hash_P_str: " << hash_P_str << '\n';

    std::cout << is_periodic(P) << '\n';  // False

    P = torch::tensor(
	 	 	 	 {{0., 0., 1.},
                  {1., 0., 0.},
                  {0., 1., 0.}}, torch::kDouble);


    std::cout << is_periodic(P) << '\n';  // True

	std::cout << "Done!\n";
	return 0;
}





