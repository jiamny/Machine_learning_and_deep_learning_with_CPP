/*
 * AdaBoost.hpp
 *
 *  Created on: May 9, 2024
 *      Author: jiamny
 */

#ifndef ALGORITHMS_ADABOOST_HPP_
#define ALGORITHMS_ADABOOST_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class AdaBoost {
public:
    //AdaBoost算法
	 AdaBoost(torch::Tensor _X, torch::Tensor _y, double _tol=0.05, int _max_iter=10) {
        X = _X;
        y = _y;
        tol = _tol;
        max_iter = _max_iter;
        w = torch::zeros({X.size(0)}, torch::kDouble);
        w.fill_(1.0/X.size(0));
        G = {};
	}

	std::tuple<std::map<std::string, double>, torch::Tensor, torch::Tensor> build_stump(void) {
		 /*
	        以带权重的分类误差最小为目标，选择最佳分类阈值
	       best_stump['dim'] 合适的特征所在维度
	       best_stump['thresh']  合适特征的阈值
	       best_stump['ineq']  树桩分类的标识lt,rt
	      */
		 int64_t m = X.size(0), n = X.size(1);
		 // 分类误差
		 double e_min = std::numeric_limits<double>::infinity();
		 // 小于分类阈值的样本属于的标签类别
		 torch::Tensor sign;
		 // 最优分类树桩
		 std::map<std::string, double> best_stump;

		 for(int i =0; i < n; i++) {
	    	 torch::Tensor range_min = X.index({Slice(), i}).min();  // 求每一种特征的最大最小值
	    	 torch::Tensor range_max = X.index({Slice(), i}).max(); //   range_max = self.X[:, i].max()
	    	 double step_size = (range_max.data().item<double>() - range_min.data().item<double>()) / n;

	         for(int j = -1; j < (n + 1); j++) {// in range(-1, int(n) + 1):
	        	 double thresh_val = range_min.data().item<double>() + j * step_size;
	             // 计算左子树和右子树的误差
	             for(auto& inequal : inequals) {
	            	 torch::Tensor predict_vals = base_estimator(X, i, thresh_val,
	                                                       inequal);
	            	 torch::Tensor err_arr = torch::ones({m}, torch::kDouble);
	                 err_arr.masked_fill_(predict_vals.t() == y.t(), 0.0);
	                 double weighted_error = torch::dot(w, err_arr).data().item<double>();
	                 if( weighted_error < e_min ) {
	                     e_min = weighted_error;
	                     sign = predict_vals;
	                     best_stump["dim"] = static_cast<double>(i);
	                     best_stump["thresh"] = static_cast<double>(thresh_val);
	                     best_stump["ineq"] = static_cast<double>(inequal);
	                 }
	             }
	         }
	     }
	     return std::make_tuple(best_stump, sign, torch::tensor({e_min}, torch::kDouble));
	 }

    void updata_w(double alpha, torch::Tensor predict) {
        //更新样本权重w
        // 以下2行根据公式8.4 8.5 更新样本权重
        torch::Tensor P = w * torch::exp(-1.*alpha * y * predict);
        w = P / P.sum();
    }

    torch::Tensor base_estimator(torch::Tensor X, int dimen, double threshVal, double threshIneq) {
        // 计算单个弱分类器（决策树桩）预测输出

    	torch::Tensor ret_array = torch::ones({X.size(0)}, torch::kFloat32);  // 预测矩阵

        // 左叶子 ，整个矩阵的样本进行比较赋值
        if( static_cast<int>(threshIneq) == 0 )
            ret_array.masked_fill_( X.index({Slice(), dimen}) <= threshVal, -1.0);
        else
            ret_array.masked_fill_( X.index({Slice(), dimen}) > threshVal, -1.0);

        return ret_array;
    }

    void fit(void) {
    	//对训练数据进行学习
        torch::Tensor g = torch::empty(0);
        for(int i = 0; i < max_iter; i++) {
        	std::map<std::string, double> best_stump;
        	torch::Tensor sign, e_min;

            std::tie(best_stump, sign, e_min) = build_stump();  // 获取当前迭代最佳分类阈值
            double error = e_min.data().item<double>();
            error += 1e-5;   									// avoid divide by zero
            double alpha = 1.0 / 2 * std::log((1 - error) / error);     // 计算本轮弱分类器的系数
            // 弱分类器权重
            best_stump["alpha"] = alpha;
            // 保存弱分类器
            G.push_back(best_stump);

            // 以下3行计算当前总分类器（之前所有弱分类器加权和）分类效率
            if( g.numel() == 0 )
            	g = sign.clone();
            else
            	g += alpha * sign;

            torch::Tensor y_predict = torch::sign(g);
            double error_rate = torch::sum(torch::abs(y_predict - y)).data().item<double>()*1.0 / 2 / y.size(0);

            if( error_rate < tol) { // 满足中止条件 则跳出循环
                std::cout << "迭代次数: " <<  std::to_string(i + 1) << "\n";
                break;
            } else {
                updata_w(alpha, y_predict);  // 若不满足，更新权重，继续迭代
            }
        }
    }

    torch::Tensor predict(torch::Tensor X) {
         // 对新数据进行预测

        int64_t m = X.size(0);
        torch::Tensor g = torch::zeros({m}, torch::kDouble);
        for(int i = 0; i < G.size(); i++ ) {
        	std::map<std::string, double> stump = G[i];
            // 遍历每一个弱分类器，进行加权
        	torch::Tensor _G = base_estimator(X, static_cast<int>(stump["dim"]),
        			static_cast<double>(stump["thresh"]),
					static_cast<double>(stump["ineq"]));
            double alpha = static_cast<double>(stump["alpha"]);
            g += alpha * _G;
        }
        torch::Tensor y_predict = torch::sign(g);
        return y_predict.to(torch::kInt32);
    }

    double score(torch::Tensor X, torch::Tensor y) {
        //对训练效果进行评价"""
    	torch::Tensor  y_predict = predict(X);
        //error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
		double error_rate = torch::sum(torch::abs(y_predict - y)).data().item<double>()*1.0 / 2 / y.size(0);
        return 1 - error_rate;
    }

private:
	 torch::Tensor X, y, w;
	 std::vector<std::map<std::string, double>> G;
	 int max_iter;
	 std::vector<double> inequals = {0., 1.}; // 0 == lt, 1 == rt
	 double tol;
};

#endif /* ALGORITHMS_ADABOOST_HPP_ */
