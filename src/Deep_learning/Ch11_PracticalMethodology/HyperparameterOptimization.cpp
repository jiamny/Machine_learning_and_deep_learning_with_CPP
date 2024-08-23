/*
 * HyperparameterOptimization.cpp
 *
 *  Created on: Aug 11, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <random>
#include <limits.h>
#include <ctime>
#include <cstdlib>

#include "../../Algorithms/RandomForest.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/csvloader.h"
#include "../../Utils/TempHelpFunctions.h"
//#include "../../Algorithms/LogisticRegression.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

namespace F = torch::nn::functional;

torch::Tensor k(torch::Tensor xs, torch::Tensor  xt, double sigma=1) {
	// 协方差函数。
	torch::Tensor dx = torch::unsqueeze(xs, 1) - torch::unsqueeze(xt, 0);
	return torch::exp(-(dx.pow(2)) / (2.0*std::pow(sigma,2)));
}

torch::Tensor m(torch::Tensor x) {
	//均值函数。
	return torch::zeros_like(x);
}

torch::Tensor f(torch::Tensor x) {
	//目标函数。
	std::vector<double> coefs = {6., -2.5, -2.4, -0.1, 0.2, 0.03};
	torch::Tensor y = m(x);
	for(int i = 0;  i < coefs.size(); i++) {
		y += coefs[i] * (x.pow(i));
	}
	return y;
}

class MultivariateNormalx{
    torch::Tensor mean, stddev, var, L;
    int d = 0;
    // Define epsilon.
    double epsilon = 0.0001;
public:
    MultivariateNormalx(const torch::Tensor &mean, const torch::Tensor &std) : mean(mean), stddev(std), var(std * std) {
      	d = mean.size(0);
    	// Add small pertturbation.
    	torch::Tensor K = stddev + epsilon*torch::eye(d).to(torch::kDouble);
    	// Cholesky decomposition.
    	L = torch::linalg::cholesky(K);
    }

    torch::Tensor rsample(int n = 1) {
    	torch::Tensor u = torch::normal(0., 1., d*n).reshape({d, n}).to(torch::kDouble);
    	torch::Tensor x = mean + torch::mm(L, u);
    	return x;
    }
};
// Wichura, M. J. (1988).
// Algorithm AS 241: The Percentage Points of the Normal Distribution.
// Journal of the Royal Statistical Society. Series C (Applied Statistics), 37(3), 477-484.
double normal_ppf(double p, double mean = 0., double std_dev = 1.) {
    if (p < 0 || p > 1 || std_dev <= 0 || std::isnan(mean) || std::isnan(std_dev)) {
        return NAN;
    }

    if (p == 0) {
        return -INFINITY;
    }

    if (p == 1) {
        return INFINITY;
    }

    double q = p - 0.5;
    if (std::fabs(q) < 0.425) {
        double r = 0.180625 - q * q;
        return mean + std_dev * q *
            (((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r + 6.7265770927008700853e4) * r + 4.5921953931549871457e4) * r + 1.3731693765509461125e4) * r + 1.9715909503065514427e3) * r + 1.3314166789178437745e2) * r + 3.3871328727963666080e0) /
            (((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r + 3.9307895800092710610e4) * r + 2.1213794301586595867e4) * r + 5.3941960214247511077e3) * r + 6.8718700749205790830e2) * r + 4.2313330701600911252e1) * r + 1);
    } else {
        double r = q < 0 ? p : 1 - p;
        r = std::sqrt(-std::log(r));
        double sign = q < 0 ? -1 : 1;
        if (r < 5) {
            r -= 1.6;
            return mean + std_dev * sign *
                (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r + 1.27045825245236838258e0) * r + 3.64784832476320460504e0) * r + 5.76949722146069140550e0) * r + 4.63033784615654529590e0) * r + 1.42343711074968357734e0) /
                (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r + 1.67638483018380384940e0) * r + 2.05319162663775882187e0) * r + 1);
        } else {
            r -= 5;
            return mean + std_dev * sign *
                (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r + 1.78482653991729133580e0) * r + 5.46378491116411436990e0) * r + 6.65790464350110377720e0) /
                (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r + 5.99832206555887937690e-1) * r + 1);
        }
    }
}


class RBFKernel {
public:
	RBFKernel(double _sigma = -1.0) {
        // RBF 核。
        sigma = _sigma;  // 如果 sigma 未赋值则默认为 np.sqrt(n_features/2)，n_features 为特征数。
	}

	torch::Tensor _kernel(torch::Tensor X, torch::Tensor Y=torch::empty(0)) {
      /*
        对 X 和 Y 的行的每一对计算 RBF 核。如果 Y 为空，则 Y=X。

        参数说明：
        X：输入数组，为 (n_samples, n_features)
        Y：输入数组，为 (m_samples, n_features)
       */
    	if(X.dim() == 1)
    		X = X.reshape({-1, 1});

        if( Y.numel() < 1 )
        	Y = X.clone();

        if(Y.dim() == 1)
        	Y = Y.reshape({-1, 1});

        assert(X.dim() == 2);
        assert(Y.dim() == 2);						// "X and Y must have 2 dimensions"

        if( sigma <= 0.)
        	sigma = std::sqrt(X.size(1)*1.0 / 2);	//if self.params["sigma"] is None else self.params["sigma"]

        X = X / sigma;
        Y = Y / sigma;
        c10::OptionalArrayRef<long int> dim = {1};

        torch::Tensor D = (-2 * X).mm(Y.t()) + torch::sum(torch::pow(Y, 2), dim) + torch::sum(torch::pow(X,2), dim).unsqueeze(1); //[:, np.newaxis]
        //D[D < 0] = 0
        D.masked_fill_(D < 0,  0);
        return torch::exp(-0.5 * D);
    }

private:
	double sigma;
};

class GPRegression {
    //高斯过程回归
public:
	GPRegression(double _sigma=1e-10) {
        kernel = RBFKernel();
        GP_mean = torch::empty(0);
        GP_cov = torch::empty(0);
        X = torch::empty(0);
        sigma =_sigma;
	}

	void fit(torch::Tensor _X, torch::Tensor _y) {
		X = _X;
		y = _y;
		GP_mean = torch::zeros(X.size(0)).to(torch::kDouble);
		GP_cov = kernel._kernel(X, X);
	}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict(torch::Tensor X_star, double conf_interval=0.95) {
       /*
        对新的样本 X 进行预测。

        参数说明：
        X_star：输入数组，为 (n_samples, n_features)
        conf_interval：置信区间，浮点型 (0, 1)，default=0.95
	   */
		torch::Tensor K = GP_cov.clone();
		torch::Tensor K_star = kernel._kernel(X_star, X);
		torch::Tensor K_star_star = kernel._kernel(X_star, X_star);
		torch::Tensor sig = torch::eye(K.size(0)).to(torch::kDouble) * sigma;

		torch::Tensor K_y_inv = torch::linalg::pinv(K + sig);
		torch::Tensor mean = K_star.mm(K_y_inv).mm(y);
		torch::Tensor cov = K_star_star - K_star.mm(K_y_inv).mm(K_star.t());

        double percentile = normal_ppf(conf_interval);
        torch::Tensor conf = percentile * torch::sqrt(torch::diag(cov));
        return std::make_tuple(mean, conf, cov);
	}

private:
	RBFKernel kernel;
	torch::Tensor GP_mean, GP_cov, X, y, mu;
	double sigma;
};

double func_black_box(int k, torch::Tensor X_train, torch::Tensor y_train, torch::Tensor X_test,torch::Tensor y_test) {
	RandomForest model = RandomForest(k, 5, 30);
    model.fit(X_train, y_train);
    return model.score(X_test, y_test);
}

class BayesianOptimization {
public:
	BayesianOptimization() {
        model = GPRegression();
	}

	torch::Tensor acquisition_function(torch::Tensor Xsamples) {
    	torch::Tensor mu, cov, _;
        std::tie(mu, _, cov) = model.predict(Xsamples);

        MultivariateNormalx mvn(mu, cov);
        torch::Tensor ysample = mvn.rsample().to(torch::kDouble);
        return ysample;
    }

	torch::Tensor opt_acquisition(torch::Tensor X, int n_samples=20) {
        // 样本搜索策略，一般方法有随机搜索、基于网格的搜索，或局部搜索
        // 我们这里就用简单的随机搜索，这里也可以定义样本的范围
    	torch::Tensor Xsamples = torch::randint(1, 50, {n_samples, X.size(1)}).to(torch::kDouble);

        // 计算采集函数的值并取最大的值
    	torch::Tensor scores = acquisition_function(Xsamples);
    	int ix = torch::argmax(scores).data().item<int>();
        return Xsamples[ix][0];
    }

	std::tuple<torch::Tensor, torch::Tensor> fit(torch::Tensor X, torch::Tensor y,
			torch::Tensor X_train, torch::Tensor y_train, torch::Tensor X_test,torch::Tensor y_test) {
    	torch::Tensor mean, conf, cov;
        // 拟合 GPR 模型
        model.fit(X, y);

        // 优化过程
        for(auto& i : range(15, 0)) {
        	int x_star = opt_acquisition(X).data().item<int>();	// 下一个采样点
            double y_star = func_black_box(x_star, X_train, y_train, X_test, y_test);
            std::tie(mean, conf, cov) = model.predict( torch::tensor({{x_star}}).to(torch::kDouble) );
            // 添加当前数据到数据集合
            X = torch::cat({X, torch::tensor({{x_star}}).to(torch::kDouble)}, 0);
            y = torch::cat({y, torch::tensor({{y_star}}).to(torch::kDouble)}, 0);

            // 更新 GPR 模型
            model.fit(X, y);
            printf("Fitting iteration %3d\n", (i + 1));
        }
        int ix = torch::argmax(y.squeeze()).data().item<int>();
        printf("Best Result: x=%.2f, y=%.4f", X[ix].data().item<double>(), y[ix].data().item<double>());
        return std::make_tuple(X[ix], y[ix]);
    }

private:
	GPRegression model;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Sampling\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor x = torch::tensor({-4., -1.5, 0., 1.5, 2.5, 2.7}, torch::kDouble);
	torch::Tensor y = f(x);
	printVector(tensorTovector(y));

	torch::Tensor x_star = torch::tensor(linspace(-8, 8, 100), torch::kDouble);
	torch::Tensor K = k(x, x);
	std::cout << K << '\n';
	torch::Tensor K_star = k(x, x_star);
	torch::Tensor K_star_star = k(x_star, x_star);
	y = y.reshape({y.size(0),-1});
	torch::Tensor mu = m(x_star.unsqueeze(1)) + (K_star.t().mm(torch::linalg::pinv(K)).mm(y-m(y)));
	printVector(tensorTovector(mu.squeeze()));
	torch::Tensor Sigma = K_star_star - K_star.t().mm(torch::linalg::pinv(K)).mm(K_star);
	std::cout <<"mu: " << mu.sizes() << " " << Sigma.sizes() << '\n';
	torch::Tensor y_true = f(x_star);
	printVector(tensorTovector(y_true));

	MultivariateNormalx mvn(mu, Sigma);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	auto s = matplot::scatter(ax, tensorTovector(x), tensorTovector(y), 20.0);
	s->marker_color("b");
	s->marker_face_color({0, .5, .5});
	s->display_name("Training data");
	matplot::plot(ax, tensorTovector(x_star.squeeze()),
			tensorTovector(y_true.squeeze()), "b--")->line_width(3).display_name("True f(x)");
	std::vector<std::string> Colors = {"c", "r", "g", "m"};
	for(auto& c : Colors) {
		torch::Tensor y_star = mvn.rsample().to(torch::kDouble);
	    if(c == "c")
	    	matplot::plot(ax, tensorTovector(x_star.squeeze()),
	    			tensorTovector(y_star.squeeze()), "c")->line_width(2).display_name("c data");
	    if(c == "r")
	    	matplot::plot(ax, tensorTovector(x_star.squeeze()),
	    			tensorTovector(y_star.squeeze()), "r")->line_width(2).display_name("r data");
	    if(c == "g")
	    	matplot::plot(ax, tensorTovector(x_star.squeeze()),
	    			tensorTovector(y_star.squeeze()), "g")->line_width(2).display_name("g data");
	    if(c == "m")
	    	matplot::plot(ax, tensorTovector(x_star.squeeze()),
	    			tensorTovector(y_star.squeeze()), "m")->line_width(2).display_name("m data");
	}
	matplot::ylim(ax, {-8, 8});
	matplot::plot(ax, tensorTovector(x_star.squeeze()),
			tensorTovector(mu.squeeze()), "k")->line_width(2).display_name("Mean");
	matplot::legend(ax, {});
	matplot::hold(ax, false);
	matplot::show();

	std::cout << normal_ppf(0.95, 0., 1.) << '\n';
	torch::manual_seed(321);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";
	std::ifstream file;
	std::string path = "./data/breast_cancer_wisconsin.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records -= 1; 		// if first row is heads but not record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));
	printVector(indices);
	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.75);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> iMap;
	std::cout << "iMap.empty(): " << iMap.empty() << '\n';

	// zscore = true, normalize data = true
	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, true, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	// set random number generator seed
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Set random number generator seed for random_choice()\n";
	std::cout << "// --------------------------------------------------\n";
	std::srand((unsigned) time(NULL));

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Bayesian optimization with gaussian process regression\n";
	std::cout << "// --------------------------------------------------\n";

	int n_estimators = 4, min_features = 5, max_depth = 30;
    RandomForest model = RandomForest(n_estimators, min_features, max_depth);
    model.fit(train_dt, train_lab);
    double ds = model.score( test_dt, test_lab);
    printf("\ninitial n_estimators=%d, score=%.4f\n", n_estimators, ds);

    torch::Tensor X0 = torch::tensor({{n_estimators}}).to(torch::kDouble);
    torch::Tensor y0 = torch::tensor({{ds}}).to(torch::kDouble);
    BayesianOptimization B0 = BayesianOptimization();

    torch::Tensor best_n_estimators, best_score;
    std::tie(best_n_estimators, best_score) = B0.fit(X0, y0, train_dt, train_lab, test_dt, test_lab);
    printf("\nBest n_estimators=[%d], score=[%.4f]\n", static_cast<int>(best_n_estimators.data().item<double>()),
    		best_score.data().item<double>());

	std::cout << "Done!\n";
	return 0;
}



