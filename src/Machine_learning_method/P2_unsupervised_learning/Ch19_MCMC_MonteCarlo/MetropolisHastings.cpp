/*
 * MetropolisHastings.cpp
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

double binom(int n, int k) { return 1/((n+1)*std::beta(n-k+1,k+1)); }

double binom_pmf(int n, int k, double x) {
	return binom(n, k)*std::pow(x, k)*std::pow(1.0 - x, n - k);
}

template <typename RealType = double>
class beta_distribution
{
  public:
    typedef RealType result_type;

    class param_type
    {
      public:
        typedef beta_distribution distribution_type;

        explicit param_type(RealType a = 2.0, RealType b = 2.0)
          : a_param(a), b_param(b) { }

        RealType a() const { return a_param; }
        RealType b() const { return b_param; }

        bool operator==(const param_type& other) const
        {
          return (a_param == other.a_param &&
                  b_param == other.b_param);
        }

        bool operator!=(const param_type& other) const
        {
          return !(*this == other);
        }

      private:
        RealType a_param, b_param;
    };

    explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
      : a_gamma(a), b_gamma(b) { }
    explicit beta_distribution(const param_type& param)
      : a_gamma(param.a()), b_gamma(param.b()) { }

    void reset() { }

    param_type param() const
    {
      return param_type(a(), b());
    }

    void param(const param_type& param)
    {
      a_gamma = gamma_dist_type(param.a());
      b_gamma = gamma_dist_type(param.b());
    }

    template <typename URNG>
    result_type operator()(URNG& engine)
    {
      return generate(engine, a_gamma, b_gamma);
    }

    template <typename URNG>
    result_type operator()(URNG& engine, const param_type& param)
    {
      gamma_dist_type a_param_gamma(param.a()),
                      b_param_gamma(param.b());
      return generate(engine, a_param_gamma, b_param_gamma);
    }

    result_type min() const { return 0.0; }
    result_type max() const { return 1.0; }

    result_type a() const { return a_gamma.alpha(); }
    result_type b() const { return b_gamma.alpha(); }

    bool operator==(const beta_distribution<result_type>& other) const
    {
      return (param() == other.param() &&
              a_gamma == other.a_gamma &&
              b_gamma == other.b_gamma);
    }

    bool operator!=(const beta_distribution<result_type>& other) const
    {
      return !(*this == other);
    }

    template <typename URNG>
    result_type pdf(URNG& x)
    {
    	result_type a = a_gamma.alpha();
    	result_type b = b_gamma.alpha();
        return (1.0/std::beta(a, b))*std::pow(x, a - 1)*std::pow(1. - x, b - 1);
    }

  private:
    typedef std::gamma_distribution<result_type> gamma_dist_type;

    gamma_dist_type a_gamma, b_gamma;

    template <typename URNG>
    result_type generate(URNG& engine,
      gamma_dist_type& x_gamma,
      gamma_dist_type& y_gamma)
    {
      result_type x = x_gamma(engine);
      return x / (x + y_gamma(engine));
    }
};


class AcceptedDistribution {
    //接受分布
public:

    static double prob(double x) {
        //P(X = x) 的概率
        // Bin(4, 10)
    	int n = 10, k = 4;
        return binom_pmf(n, k, x); //binom.pmf(4, 10, x)
    }
};

class ProposalDistribution {
    //建议分布
public:
	static double sample() {
        //从建议分布中抽取一个样本
		//B(1,1)
		double a = 1.0, b = 1.0;
		std::random_device rd;
		std::mt19937 gen(rd());
		beta_distribution<> beta(a, b);
        return beta(gen);
    }

	static double prob(double x) {
        //P(X = x) 的概率
		double a = 1.0, b = 1.0;
		beta_distribution<> beta(a, b);
        return beta.pdf(x);
	}

    static double joint_prob(double  x_1, double x_2) {
        //P(X = x_1, Y = x_2) 的联合概率
        return prob(x_1) * prob(x_2);
    }
};

class MetropolisHastings {
public:
	MetropolisHastings(int _m=1e4, int _n=1e5) {
        /*
        Metropolis Hastings 算法
        :param m: 收敛步数
        :param n: 迭代步数
        */
        //self.proposal_dist = proposal_dist
        //self.accepted_dist = accepted_dist
        m = _m;
        n = _n;
	}

    double __calc_acceptance_ratio(double x, double x_prime) {
       /*
        计算接受概率
        :param x: 上一状态
        :param x_prime: 候选状态
       */
        double prob_1 = AcceptedDistribution::prob(x_prime) * ProposalDistribution::joint_prob(x_prime, x);
        double prob_2 = AcceptedDistribution::prob(x) * ProposalDistribution::joint_prob(x, x_prime);
        double alpha = std::min(1., (prob_1 / prob_2));
        return alpha;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> solve(void) {
        // Metropolis Hastings 算法求解

        torch::Tensor all_samples = torch::zeros({n}, torch::kDouble);
        // (1) 任意选择一个初始值
        double x_0 = torch::rand({1}, torch::kDouble).data().item<double>();
        // (2) 循环执行
        for(auto& i : range(n, 0)) {
            double x = x_0;
            if(i != 0)
            	x = all_samples[i - 1].data().item<double>();

            // (2.a) 从建议分布中抽样选取
            double x_prime = ProposalDistribution::sample();
            //#print('x_prime: ', x_prime)
            // (2.b) 计算接受概率
            double alpha = __calc_acceptance_ratio(x, x_prime);
            // (2.c) 从区间 (0,1) 中按均匀分布随机抽取一个数 u
            torch::Tensor u = torch::zeros({1}, torch::kDouble);
            u.uniform_(0, 1);
            // 根据 u <= alpha，选择 x 或 x_prime 进行赋值
            if( u.data().item<double>() <= alpha )
                all_samples[i] = x_prime;
            else
                all_samples[i] = x;
        }
        // (3) 随机样本集合
        torch::Tensor samples = all_samples.index({Slice(m, None)}); //[self.m:]
        // 函数样本均值
        torch::Tensor dist_mean = samples.mean();
        // 函数样本方差
        torch::Tensor dist_var = samples.var();
        return std::make_tuple(samples, dist_mean, dist_var);
    }

    static void visualize(torch::Tensor samples, size_t bins=50) {
        /*
          可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        */

    	auto F = figure(true);
    	F->size(800, 600);
    	F->add_axes(false);
    	F->reactive_mode(false);
    	F->tiledlayout(1, 1);
    	F->position(0, 0);

    	auto ax = F->nexttile();

    	std::vector<double> d = tensorTovector(samples.to(torch::kDouble));
    	matplot::hist(ax, d)->num_bins(bins).display_name("Samples Distribution");
        xlim(ax, {0, 1});
        //legend({});
		title(ax, "Metropolis Hastings");
		show();
    }
private:
    int m = 0, n = 0;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

    for (int i = 0; i < 5; i++) {
      std::cout << ProposalDistribution::sample() << std::endl;
    }
    std::cout << ProposalDistribution::prob(0.7870827539519615) << '\n';

    // 收敛步数
    int m = 1000;
	// 迭代步数
    int n = 10000;

    MetropolisHastings metropolis_hastings(m, n);

	// 使用 Metropolis-Hastings 算法进行求解
    torch::Tensor samples, dist_mean, dist_var;
    std::tie(samples, dist_mean, dist_var) = metropolis_hastings.solve();
    std::cout << "均值: " << dist_mean.data().item<double>() << '\n';
	std::cout << "方差: " << dist_var.data().item<double>() << '\n';

	// 对结果进行可视化
	MetropolisHastings::visualize(samples, 20);

	std::cout << "Done!\n";
	return 0;
}
