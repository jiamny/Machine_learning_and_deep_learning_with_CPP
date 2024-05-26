/*
 * NaiveBayes.hpp
 *
 *  Created on: May 4, 2024
 *      Author: jiamny
 */

#ifndef NAIVEBAYES_HPP_
#define NAIVEBAYES_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>
#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class NaiveBayes {
public:
	NaiveBayes() {
		parameters.clear(); // 保存每个特征针对每个类的均值和方差
        y = torch::empty(0);
        classes = torch::empty(0);
	};

    void fit(torch::Tensor X, torch::Tensor _y) {
        y = _y;
    	classes = std::get<0>(at::_unique(y)); // 类别
    	// 计算每个特征针对每个类的均值和方差
    	for(int64_t i = 0; i < classes.size(0); i++ ) {
    	    auto c = classes[i];
    	    // 选择类别为c的X
    	    torch::Tensor ind = y.eq(c).to(torch::kLong);
    	    std::vector<int64_t> slt;
    	    for(int64_t n = 0; n < ind.size(0); n++)
    	    	if( ind[n].data().item<long>() > 0)
    	    		slt.push_back(n);
    	    	torch::Tensor Idx = torch::from_blob(
    	                    		slt.data(), {static_cast<long int>(slt.size())}, torch::kLong).clone();
    	    	torch::Tensor X_where_c = torch::index_select(X, 0, Idx );

    	    	std::vector<std::map<std::string, double>> pra;
    	    	parameters.push_back(pra);
    	    	// 添加均值与方差
    	    	for(int64_t j = 0; j < X_where_c.t().size(0); j++) {
    	    	    torch::Tensor col = X_where_c.index({j, Slice()});
    	    		std::map<std::string, double> parameter = {{"mean", col.mean().data().item<double>()},
    	    				{"var", col.var().data().item<double>()}};
    	    	    parameters[i].push_back(parameter);
    	    	}
    	}
    }

    double _calculate_prior(torch::Tensor c) {
        // 先验函数。
        auto frequency = torch::mean(y.eq(c).to(torch::kDouble));
        return frequency.data().item<double>();
    }

    double _calculate_likelihood(double mean, double var, double X) {
        // 似然函数。
        // 高斯概率
        double eps = 1e-4; // 防止除数为0
        double coeff = 1.0 / std::sqrt(2.0 * M_PI * var + eps);
        double exponent = std::exp(-1.0*(std::pow(X - mean, 2) / (2 * var + eps)));
        return coeff * exponent;
    }

    int64_t _calculate_probabilities(torch::Tensor X) {
        torch::Tensor posteriors = torch::zeros({classes.size(0)}).to(X.dtype());
        for(int64_t i = 0; i < classes.size(0); i++) {
            double posterior = _calculate_prior(classes[i]);
            std::vector<std::map<std::string, double>> cparams = parameters[i];
            for(int64_t j = 0; j < X.size(0); j++) {
                // 独立性假设
                // P(x1,x2|Y) = P(x1|Y)*P(x2|Y)
            	double feature_value = X[j].data().item<double>();
            	std::map<std::string, double> params = cparams[j];
                double likelihood = _calculate_likelihood(params["mean"], params["var"], feature_value);
                posterior *= likelihood;
            }
            posteriors.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({i})}, posterior);
        }
        // 返回具有最大后验概率的类别
        auto cidx = torch::argmax(posteriors);
        return classes[cidx.data().item<int64_t>()].data().item<int64_t>();
    }

    torch::Tensor predict(torch::Tensor X) {
    	torch::Tensor y_pred = torch::zeros({X.size(0)}).to(torch::kLong);
		//= [self._calculate_probabilities(sample) for sample in X]
    	for(int64_t i = 0; i < X.size(0); i++) {
    		torch::Tensor sample = X.index({i, Slice()});
    		int64_t p = _calculate_probabilities(sample);
    		y_pred[i].fill_(p);
    	}
        return y_pred;
    }

    double score(torch::Tensor X_ts, torch::Tensor y_ts, bool verbose = false) {
        torch::Tensor y_pred = predict(X_ts).squeeze_();
        if( verbose ) {
        	std::cout << "y_ts: " << y_ts.sizes() << '\n';
        	std::cout << "y_pred: " << y_pred.sizes() << '\n';
        	printVector(tensorTovector(y_pred.to(torch::kDouble)));
        }
        c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum(y_ts.squeeze() == y_pred, dim).data().item<double>() / X_ts.size(0);
        return accuracy;
    }

private:
	torch::Tensor y, classes;
	std::vector<std::vector<std::map<std::string, double>>> parameters;
};

#endif /* NAIVEBAYES_HPP_ */
