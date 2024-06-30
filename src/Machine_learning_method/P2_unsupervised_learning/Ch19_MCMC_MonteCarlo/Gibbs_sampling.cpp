/*
 * Gibbs_sampling.cpp
 *
 *  Created on: Jun 7, 2024
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

#include <matplot/matplot.h>
using namespace matplot;

class TargetDistribution {
    //目标概率分布
public:
	double c = 0.;
	TargetDistribution() {
        // 联合概率值过小，可对建议分布进行放缩
        c = __select_prob_scaler();
	}

    double sample(torch::Tensor x, int k=0) {

        //使用接受-拒绝方法从满条件分布中抽取新的分量 x_k
        double returnV = 0.0;
        double theta = x[0].data().item<double>(), eta = x[1].data().item<double>();
        if( k == 0 ) {
            while(true) {
            	torch::Tensor n_theta = torch::zeros({1}, torch::kDouble);
            	torch::Tensor alp = torch::zeros({1}, torch::kDouble);
            	n_theta.uniform_(0, 1.0 - eta);
                alp.uniform_(0., 1.0);

                double new_theta = n_theta.data().item<double>();
                double alpha = alp.data().item<double>();
                if( (alpha * c) < __prob({new_theta, eta})) {
                    //return new_theta;
                	returnV = new_theta;
                	break;
                }
            }
        } else if(k == 1) {
            while(true) {
            	torch::Tensor n_eta = torch::zeros({1}, torch::kDouble);
            	torch::Tensor alp = torch::zeros({1}, torch::kDouble);
            	n_eta.uniform_(0, 1.0 - theta);
            	alp.uniform_(0., 1.0);
            	double new_eta = n_eta.data().item<double>();
            	double alpha = alp.data().item<double>();
                if( (alpha * c) < __prob({theta, new_eta}) ) {
                    //return new_eta;
                	returnV = new_eta;
                	break;
                }
            }
        }
        return returnV;
    }

    double __select_prob_scaler(void) {
        //选择合适的建议分布放缩尺度

        std::vector<double> prob_list;
        double step = 1e-3;
        for(auto& theta : tensorTovector(torch::arange(step, 1, step).to(torch::kDouble))) {
            for(auto& eta : tensorTovector(torch::arange(step, 1 - theta + step, step).to(torch::kDouble))) {
                double prob = __prob({theta, eta});
                prob_list.push_back(prob);
            }
        }
        auto maxPos = std::max_element(prob_list.begin(), prob_list.end());
        double searched_max_prob = prob_list[maxPos - prob_list.begin()];
        double upper_bound_prob = searched_max_prob * 10;
        return upper_bound_prob;
    }

    //@staticmethod
    static double __prob(std::vector<double> x) {
        // P(X = x) 的概率

        double theta = x[0];
        double eta = x[1];
        double p1 = std::pow((theta / 4 + 1.0 / 8), 14);
        double p2 = theta / 4;
        double p3 = eta / 4;
        double p4 = (eta / 4 + 3.0 / 8);
        double p5 = 1.0 / 2 * std::pow((1 - theta - eta), 5);
        double p = (p1 * p2 * p3 * p4 * p5);
        return p;
    }
};


class GibbsSampling {
public:
	TargetDistribution target_dist;
	int m = 1e4, n = 1e5, j = 2;
	GibbsSampling(TargetDistribution _target_dist, int _j, int _m, int _n) {
        /*
        Gibbs Sampling 算法

        :param target_dist: 目标分布
        :param j: 变量维度
        :param m: 收敛步数
        :param n: 迭代步数
        */
        target_dist = _target_dist;
        j = _j;
        m = _m;
        n = _n;
	}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> solve(void) {
        //Gibbs Sampling 算法求解
        // (1) 初始化
        torch::Tensor all_samples = torch::zeros({n, j});
        // 任意选择一个初始值
        torch::Tensor x_0 = torch::rand({j});
        // (2) 循环执行
        for(auto& i : range(n, 0)) {
            torch::Tensor x = x_0.clone();
            if(i != 0)
            	x = all_samples[i - 1].clone();

            // 满条件分布抽取
            for(auto& k : range(j, 0)) {
                x[k] = target_dist.sample(x, k);
            }
            all_samples.index_put_({i, Slice()}, x);
        }
        // (3) 得到样本集合
        torch::Tensor samples = all_samples.index({Slice(m, None), Slice()});
        // (4) 计算函数样本均值
        c10::OptionalArrayRef<long int> dim = {0};
        torch::Tensor dist_mean = samples.mean(dim);
        torch::Tensor dist_var = samples.var(0);
        return std::make_tuple(samples, dist_mean, dist_var); // [self.m:]
    }

    //staticmethod
    static void visualize(torch::Tensor samples, size_t bins=50) {
        /*
          可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        */

    	auto F2 = figure(true);
    	F2->size(1000, 800);
    	F2->add_axes(false);
    	F2->reactive_mode(false);
    	F2->tiledlayout(1, 1);
    	F2->position(0, 0);

    	auto ax = F2->nexttile();
    	std::vector<double> d1 = tensorTovector(samples.index({Slice(), 0}).to(torch::kDouble));
    	std::vector<double> d2 = tensorTovector(samples.index({Slice(), 1}).to(torch::kDouble));
    	auto h1 = hist(ax, d1);
    	h1->num_bins(bins).display_name("θ");
        hold(ax, on);
        auto h2 = hist(ax, d2);
        h2->num_bins(bins).display_name("η");
        xlim(ax, {0, 1});
        legend(ax, {});
		title(ax, "Gibbs Sampling");
		matplot::show();
    }
};



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// 收敛步数
	int m = 1e3;
	// 迭代步数
	int n = 1e4;

	// 目标分布
	TargetDistribution target_dist = TargetDistribution();

	// 使用 Gibbs Sampling 算法进行求解
	GibbsSampling gibbs_sampling = GibbsSampling(target_dist, 2, m, n);
	torch::Tensor samples, dist_mean, dist_var;

	std::tie(samples, dist_mean, dist_var) = gibbs_sampling.solve();

	std::cout << "theta均值： " << dist_mean[0].data().item<double>()
			  << ", theta方差： " << dist_var[0].data().item<double>() << '\n';
	std::cout << "eta均值： " << dist_mean[1].data().item<double>()
			  << ", eta方差： " << dist_var[1].data().item<double>() << '\n';

	// 对结果进行可视化
	GibbsSampling::visualize(samples, 20);

	std::cout << "Done!\n";
	return 0;
}




