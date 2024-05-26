/*
 * Information_theory.cpp
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
#include <random>
#include <algorithm>
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

#include <matplot/matplot.h>
using namespace matplot;

double H(std::string sentence) {

	// 最优编码长度
	double entropy = 0.0;
	// 这里有 256 个可能的 ASCII 符号

	for(int& i : range(255, 0)) {
		char character_i = i + '0';
		double Px = 1.0*std::count(sentence.begin(), sentence.end(), character_i)/sentence.length();
		if(Px > 0)
			entropy += -1.0*Px * std::log2(Px); // 注:log 以 2 为底
	}
	return entropy;
}


torch::Tensor kl(torch::Tensor pp, torch::Tensor qq) {
	// D(P || Q)
	return torch::sum(torch::where(pp != 0, pp * torch::log(pp / qq), 0));
}

torch::Tensor norm_pdf(torch::Tensor x, double mu, double sigma) {
        return torch::exp(-1.0*torch::pow((x - mu), 2.) / (2. * std::pow(sigma, 2.))) /
               (sigma * std::sqrt(2. * pi));
}

torch::Tensor _entropy(torch::Tensor pk, torch::Tensor qk=torch::empty(0) ) {
	if(torch::sum(pk).data().item<double>() != 1.0)
		pk = pk.div(torch::sum(pk).data().item<double>());

	torch::Tensor H;
    if( qk.numel() != 0) {
    	if(torch::sum(qk).data().item<double>() != 1.0)
    		qk = qk.div(torch::sum(qk).data().item<double>());

    	H = torch::sum(pk * torch::log(pk / qk));
    } else {
    	H = -1.0*torch::sum(pk * torch::log(pk));
    }
    return H;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 香农熵; Shannon entropy\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor p = torch::linspace(1e-6,1-1e-6,100);
	torch::Tensor entropy = (p-1.0)*torch::log(1.0-p)-p*torch::log(p);
	plot(tensorTovector( p.to(torch::kDouble) ),
			tensorTovector( entropy.to(torch::kDouble) ))->line_width(2.5);
	xlabel("p");
	ylabel("Shannon entropy in nats");
	show();

	// 只用 64 个字符
	int min = 0, max = 64;
	random_device seed;//硬件生成随机数种子
	ranlux48 engine(seed());//利用种子生成随机数引擎
	uniform_int_distribution<> distrib(min, max);//设置随机数范围，并为均匀分布

	std::string simple_message = "";
	for(int i = 0; i < 500; i++) {
		int a = distrib(engine);
		char b = a + '0';
		simple_message += b;
	}
	std::cout << "simple_message: " << simple_message << '\n';
	std::cout << "H(sentence): " << H(simple_message) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// KL 散度; Kullback-Leibler (KL) divergence\n";
	std::cout << "// --------------------------------------------------\n";
	//# 测试
	std::vector<float> kp = {0.1, 0.9};
	std::vector<float> kq = {0.1, 0.9};
	torch::Tensor pp = torch::from_blob(kp.data(), {int(kp.size())}, at::TensorOptions(torch::kFloat32)).clone();
	torch::Tensor qq = torch::from_blob(kq.data(), {int(kq.size())}, at::TensorOptions(torch::kFloat32)).clone();
	std::cout << kl(pp, qq) << '\n';

	// # D(P||Q) 与 D(Q||P) 比较
	torch::Tensor kx = torch::linspace(1, 8, 500);
	torch::Tensor ky1 = norm_pdf(kx, 3.0, 0.5);
	torch::Tensor ky2 = norm_pdf(kx, 6.0, 0.5);
	torch::Tensor kpp = ky1 + ky2;

	double mu =  10.0, sigma =  5.0;
	torch::Tensor kqq = norm_pdf(kx, mu, sigma);

	std::cout << "_entropy(kpp, kqq): " <<  _entropy(kpp, kqq) << '\n';
	std::cout << "_entropy(kqq, kpp): " <<  _entropy(kqq, kpp) << '\n';

	//# 构造 p(x)
	std::vector<double> KL_pq, KL_qp;
	std::vector<torch::Tensor> q_list;
	torch::Tensor ms = torch::linspace(0, 10, 50);
	torch::Tensor gs = torch::linspace(0.1, 5, 50);
	//# 寻找最优 q(x)
	for(int i = 0; i <  ms.size(0); i++ ) {
		double mu = ms[i].data().item<double>();
		for(int j = 0; j < gs.size(0); j++  ) {
			double sigma = gs[j].data().item<double>();
			auto kqq = norm_pdf(kx, mu, sigma);
			q_list.push_back(kqq);
			double tpq = _entropy(kpp, kqq).data().item<double>();
			double tqp = _entropy(kqq, kpp).data().item<double>();
			//std::cout << "mu: " << mu << " sigma: " << sigma
			//		  <<  " tpq: " << tpq <<  " tqp: " << tqp << '\n';
			KL_pq.push_back(tpq);
			KL_qp.push_back(tqp);
		}
	}

	std::cout << "KL_pq:\n";
	printVector(KL_pq);
	std::cout << "KL_qp:\n";
	printVector(KL_qp);

	size_t KL_pq_min = argMin(KL_pq);
	size_t KL_qp_min = argMin(KL_qp);
	std::cout << "KL_pq_min: " << KL_pq_min << " KL_pq[KL_pq_min]: " << KL_pq[KL_pq_min] << '\n';
	std::cout << "KL_qp_min: " << KL_qp_min << " KL_qp[KL_qp_min]: " << KL_qp[KL_qp_min] << '\n';

	std::vector<double> fx = tensorTovector( kx.to(torch::kDouble) );
	std::vector<double> fy = tensorTovector( (kpp/2.0).to(torch::kDouble) );
	std::vector<double> ypq = tensorTovector( q_list[KL_pq_min].to(torch::kDouble) );
	std::vector<double> yqp = tensorTovector( q_list[KL_qp_min].to(torch::kDouble) );

	auto F = figure(true);
	F->size(1200, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 2);
	F->position(0, 0);

    auto ax1 = F->nexttile();
    plot(ax1, fx, fy)->line_width(2.5).display_name("p(x)");
    ylim(ax1, {0.0, 0.8});
    hold(ax1, on);
    plot(ax1, fx, ypq, "g--")->line_width(2.5).display_name("q^*(x)");
    xlabel(ax1,"x");
    ylabel(ax1, "p(x)");
    title(ax1, "q^*= {arg\\min}_ q D_{KL}(p||q)");
    legend(ax1, {});

    auto ax2 = F->nexttile();
    plot(ax2, fx, fy, "b")->line_width(2.5).display_name("p(x)");
    ylim(ax2, {0.0, 0.8});
    hold(ax2, on);
    plot(ax2, fx, yqp, "g--")->line_width(2.5).display_name("q^*(x)");
    xlabel(ax2,"x");
    ylabel(ax2, "p(x)");
    title(ax2, "q^*= {arg\\min}_ q D_{KL}(q||p)");
    legend(ax2, {});
    show();

	std::cout << "Done!\n";
	return 0;
}


