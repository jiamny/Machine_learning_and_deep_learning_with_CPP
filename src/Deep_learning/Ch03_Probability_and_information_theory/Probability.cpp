/*
 * Probability_and_information_theory.cpp
 *
 *  Created on: Apr 23, 2024
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

#include <matplot/matplot.h>
using namespace matplot;

std::vector<double> trials(int n_samples, torch::Tensor x) {
	std::vector<double> prob;
	int samples = 0;
	for(int i = 0; i < x.size(0); i++) {
		if( x[i].data().item<int>() > 0 )
			samples += 1;
	}
	std::cout << "samples: " << samples << '\n';
	double proba_zero = (n_samples-samples)*1.0/n_samples;
	double proba_one = samples*1.0/n_samples;
	prob.push_back(proba_zero);
	prob.push_back(proba_one);
	return prob;
}

torch::Tensor k_possibilities(int k) {
    // 随机产生一组 10 维概率向量
    torch::Tensor res = torch::rand({k});
    auto _sum = torch::sum(res);
    res = res.div( _sum );
    return res;
}

std::pair<std::vector<double>, std::vector<double>> multinoulli_data(int k, int n_samples) {
	torch::Tensor  fair_probs = k_possibilities(k);
	auto sample = torch::multinomial(fair_probs, n_samples, true);

	std::vector<double> mx, my;
	for(int i = 0; i < k; i++){
		mx.push_back(i*1.0);
		my.push_back(0.0);
	}

	std::vector<double> ms = tensorTovector( sample.to(torch::kDouble) );
	for(int i = 0; i < ms.size(); i++) {
		my[ms[i]] += 1;
	}

	return std::make_pair(mx, my);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 概率分布; probability distributions\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor t = torch::zeros({1000});
	auto unid = torch::uniform(t, 0.0, 1.0);

	std::vector<double> r = tensorTovector( unid.to(torch::kDouble) );

	enum histogram::binning_algorithm alg =
	        histogram::binning_algorithm::automatic;

	const size_t n_bins = 100;

	// # 均匀分布 pdf
	auto h = hist(r, alg, histogram::normalization::pdf);
	h->num_bins(n_bins);

	hold(on);
    std::vector<double> x = linspace(0.0, 1.0);
    std::vector<double> y = transform(x, [](auto x) { return 1.0; });
    plot(x, y)->color("r").line_width(5.0f);

	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 随机变量的度量; measure random variables\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor xx = torch::tensor({1,2,3,4,5,6,7,8,9}).to(torch::kDouble);
	torch::Tensor yy = torch::tensor({9,8,7,6,5,4,3,2,1}).to(torch::kDouble);
	auto Mean = torch::mean(xx);
	// 样本方差（无偏方差）
	auto Var = torch::var(xx);
	auto zz = torch::cat({xx.reshape({1,-1}), yy.reshape({1,-1})}, 0);
	std::cout << "zz:\n" << zz << '\n';
	auto Cov = torch::cov(zz);
	std::cout << "mean:\n" << Mean << '\n';
	std::cout << "样本方差（无偏方差) var:\n" << Var << '\n'; // 样本方差（无偏方差）
	std::cout << "cov:\n" << Cov << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 常用概率分布; common probability distributions\n";
	std::cout << "// --------------------------------------------------\n";

	double p = 0.3;
	t = torch::zeros({1000});
	auto X = torch::bernoulli(t, p); // 伯努利分布
	//std::cout << "X:\n" << X << '\n';

	r = tensorTovector( unid.to(torch::kDouble) );

    auto f = figure(true);
    f->size(1200, 600);
    f->add_axes(false);
    f->reactive_mode(false);
    f->tiledlayout(1, 2);
    f->position(0, 0);

    auto ax1 = f->nexttile();
	auto f1 = hist(ax1, r, alg, histogram::normalization::pdf);
	f1->display_name("pdf");
	::matplot::legend({});

	auto ax2 = f->nexttile();
	auto f2 = hist(ax2, r, alg, histogram::normalization::cdf);
	f2->display_name("cdf");
	::matplot::legend({});

	f->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// bernoulli vs binomial\n";
	std::cout << "// --------------------------------------------------\n";
	// # 产生成功的概率
	double possibility = 0.3;
	torch::Tensor b = torch::ones({1});
	torch::Tensor pb = torch::tensor({possibility});
	torch::Tensor be = torch::binomial(b, pb);

	std::vector<double> xbe = {0, 1};
	std::vector<double> ybe = trials( 1, be);

	torch::Tensor cbi = torch::ones({1000});
	torch::Tensor bi = torch::binomial(cbi, pb);
	std::vector<double> ybi = trials( 1000, bi);

    auto fb = figure(true);
    fb->size(1200, 600);
    fb->add_axes(false);
    fb->reactive_mode(false);
    fb->tiledlayout(1, 2);
    fb->position(0, 0);

    auto axbe = fb->nexttile();
	auto fbbe = bar(axbe, xbe, ybe);
	fbbe->display_name("Bernoulli");
	::matplot::legend({});

	auto axbi = fb->nexttile();
	auto fbbi = bar(axbi, xbe, ybi);
	fbbi->display_name("Binomial");
	::matplot::legend({});

	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 范畴分布; multinoulli distribution\n";
	std::cout << "// --------------------------------------------------\n";

	int k = 10, n_samples = 1;

	std::vector<double> mx, my;
	std::tie(mx, my) = multinoulli_data(k, n_samples);

	n_samples = 1000;
	std::vector<double> mxi, myi;
	std::tie(mxi, myi) = multinoulli_data(k, n_samples);

	auto fbi = figure(true);
	fbi->size(1200, 600);
	fbi->add_axes(false);
	fbi->reactive_mode(false);
	fbi->tiledlayout(1, 2);
	fbi->position(0, 0);

    auto axmbe = fbi->nexttile();
	auto fbmbe = bar(axmbe, mx, my);
	fbmbe->display_name("Multinoulli");
	::matplot::legend({});

	auto axmbi = fbi->nexttile();
	auto fbmbi = bar(axmbi, mxi, myi);
	fbmbi->display_name("Multinomial");
	::matplot::legend({});

	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// ⾼斯分布; Gaussian distribution\n";
	std::cout << "// --------------------------------------------------\n";
	double mu = 0.0, sigma = 1.0;
	torch::Tensor N = torch::normal(mu, sigma, {100}); // # 标准正态分布
	std::cout << "N:\n" << N << '\n';;

	r = tensorTovector( N.to(torch::kDouble) );
	auto fn = figure(true);
	fn->size(1200, 600);
	fn->add_axes(false);
	fn->reactive_mode(false);
	fn->tiledlayout(1, 2);
	fn->position(0, 0);

    auto axn1 = fn->nexttile();
	auto fn1 = hist(axn1, r, alg, histogram::normalization::pdf);
	fn1->display_name("pdf");
	hold(on);
    auto pf = [&](double y) {
        return exp(-pow((y - mu), 2.) / (2. * pow(sigma, 2.))) /
               (sigma * sqrt(2. * pi));
    };
    fplot(axn1, pf, std::array<double, 2>{-3, 3})->line_width(1.5);
	::matplot::legend({"pdf", ""});

	auto axn2 = fn->nexttile();
	auto fn2 = hist(axn2, r, alg, histogram::normalization::cdf);
	fn2->display_name("cdf");
	fn2->edge_color("r");
	::matplot::legend({});

	fn->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 多元正态分布; Multivariate normal distribution\n";
	std::cout << "// --------------------------------------------------\n";
	const torch::Tensor mmu = torch::tensor({0.5, -0.2}); // 均值
	const torch::Tensor msigma = torch::tensor({{2.0, 0.3}, {0.3, 0.5}}); // 协方差矩阵
	torch::Tensor MN = at::normal(mmu, msigma);
	std::cout << "MN = \n" << MN << std::endl;

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 指数分布; Exponential distribution\n";
	std::cout << "// --------------------------------------------------\n";
	// # 定义 scale = 1 / lambda
	double lambd = 1.0;
	torch::Tensor yp = torch::zeros({1000});
	torch::Tensor xp = yp.exponential_( lambd );
	//std::cout << "xp = \n" << xp << std::endl;

	r = tensorTovector( xp.to(torch::kDouble) );
	auto fp = figure(true);
	fp->size(1200, 600);
	fp->add_axes(false);
	fp->reactive_mode(false);
	fp->tiledlayout(1, 2);
	fp->position(0, 0);

    auto axp1 = fp->nexttile();
	auto fp1 = hist(axp1, r, alg, histogram::normalization::pdf);
	fp1->display_name("pdf");
	::matplot::legend({"pdf", ""});

	auto axp2 = fp->nexttile();
	auto fp2 = hist(axp2, r, alg, histogram::normalization::cdf);
	fp2->display_name("cdf");
	fp2->edge_color("r");
	::matplot::legend({});
	fp->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 拉普拉斯分布; Laplace distribution\n";
	std::cout << "// --------------------------------------------------\n";
	double Lmu = 0.0, gamma = 1.0;
	torch::Tensor Lx = torch::linspace(-10, 10, 100);
	torch::Tensor l = (1.0/2*gamma)*torch::exp(-1.0*(torch::abs(Lx - Lmu)/gamma));
	std::vector<double> rlx = tensorTovector( Lx.to(torch::kDouble) );
	std::vector<double> rly = tensorTovector( l.to(torch::kDouble) );

	//cdf
	torch::Tensor LLx = torch::linspace(-10, 0, 50);
	torch::Tensor LRx = torch::linspace(0, 10, 50);
	torch::Tensor lL = (1.0/2*gamma)*torch::exp((LLx - Lmu)/gamma);
	torch::Tensor lR = 1.0 - (1.0/2*gamma)*torch::exp(-1.0*(LRx - Lmu)/gamma);
	std::vector<double> llx = tensorTovector( LLx.to(torch::kDouble) );
	std::vector<double> lrx = tensorTovector( LRx.to(torch::kDouble) );
	std::vector<double> lcdf = tensorTovector( lL.to(torch::kDouble) );
	std::vector<double> rcdf = tensorTovector( lR.to(torch::kDouble) );

	auto fl = figure(true);
	fl->size(1200, 450);
	fl->add_axes(false);
	fl->reactive_mode(false);
	fl->tiledlayout(1, 2);
	fl->position(0, 0);

    auto axl1 = fl->nexttile();
	auto fl1 = plot(rlx, rly);
	fl1->line_width(2.5).display_name("pdf");
	::matplot::legend({});

	auto axl2 = fl->nexttile();
	plot(axl2, llx, lcdf, "b-")->line_width(2.5);
	hold(axl2, on);
	auto fl2 = plot(axl2, lrx, rcdf, "b-");
	fl2->line_width(2.5).display_name("cdf");
	::matplot::legend({"", "cdf"});
	fl->draw();
	show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// sigmoid and softplus\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor s = torch::linspace(-10, 10, 100);
	torch::Tensor sig = Sigmoid(s);
	torch::Tensor softplus = torch::log(1.0 + torch::exp(s));
	//std::cout << "sigmoid = \n" << sigmoid << std::endl;
	//std::cout << "softplus = \n" << softplus << std::endl;

	std::vector<double> rx = tensorTovector( s.to(torch::kDouble) );
	std::vector<double> rsi = tensorTovector( sig.to(torch::kDouble) );
	std::vector<double> rso = tensorTovector( softplus.to(torch::kDouble) );

	auto fs = figure(true);
	fs->size(1200, 450);
	fs->add_axes(false);
	fs->reactive_mode(false);
	fs->tiledlayout(1, 2);
	fs->position(0, 0);

    auto axs1 = fs->nexttile();
	auto fs1 = plot(rx, rsi);
	fs1->line_width(2.5).display_name("sigmoid");
	::matplot::legend({});

	auto axs2 = fs->nexttile();
	auto fs2 = plot(rx, rso);
	fs2->line_width(2.5).display_name("softplus");
	::matplot::legend({});
	fs->draw();
	show();

	std::cout << "Done!\n";
	return 0;
}



