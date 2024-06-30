/*
 * Bagging.h
 *
 *  Created on: May 31, 2024
 *      Author: jiamny
 */

#ifndef BAGGING_H_
#define BAGGING_H_
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"
#include "RandomForest.h"

using torch::indexing::Slice;
using torch::indexing::None;

class Bagging {
    // Bagging 分类器。使用一组分类树，这些分类树使用特征训练数据的随机子集。
public:
	int n_estimators = 0, max_depth=0;
	std::vector<DecisionTree_CART> trees;

	Bagging(int _n_estimators, int _max_depth=30) {
        n_estimators = _n_estimators;			// 树的数目
        max_depth = _max_depth;					// 树的最大深度

        // 初始化决策树
        trees.clear();
        for(auto& _ : range(n_estimators, 0)) {
            trees.push_back(
            		DecisionTree_CART(max_depth));
        }
	}

    void fit(torch::Tensor X, torch::Tensor y) {
        int n_features = X.size(1);

        std::vector<std::pair<torch::Tensor, torch::Tensor>>
    	subsets = get_random_subsets(X, y, n_estimators);

        std::vector<int> samples = range(n_features, 0);

        for(auto& i : range(n_estimators, 0)) {
        	torch::Tensor X_subset = subsets[i].first;
        	torch::Tensor y_subset = subsets[i].second;

            // 用特征子集和真实值训练一棵子模型 (这里的数据也是训练数据集的随机子集)
            trees[i].fit(X_subset, y_subset);
        }
    }

    torch::Tensor predict(torch::Tensor X) {
    	torch::Tensor y_preds = torch::zeros({X.size(0), static_cast<int64_t>(trees.size())}, torch::kInt32);
        // 每棵决策树都在数据上预测
        for( int i = 0; i < trees.size(); i++ ) {
        	// 每棵决策树都在数据上预测
        	DecisionTree_CART tree = trees[i];

        	std::vector<int> prediction = tree.predict(X);
        	torch::Tensor pred =  torch::from_blob(prediction.data(), {static_cast<int64_t>(prediction.size())},
    				at::TensorOptions(torch::kInt32)).clone();
        	printf("Tree %3d predicted label: ", (i+1));
        	printVector(tensorTovector(pred.to(torch::kDouble)));

            y_preds.index_put_({Slice(), i}, pred);
        }

        torch::Tensor y_pred = torch::zeros({y_preds.size(0)}, torch::kInt32);
        // 对每个样本，选择最常见的类别作为预测
        for(int i = 0; i < y_preds.size(0); i++) {
        	torch::Tensor sample_predictions = y_preds.index({i, Slice()});
        	double d = mostFrequent(tensorTovector(sample_predictions.to(torch::kDouble)));
            y_pred.index_put_({i}, static_cast<int>(d));
        }
        return y_pred;
    }

    double score(torch::Tensor X, torch::Tensor y) {
    	torch::Tensor y_pred = predict(X);
    	c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum( y.squeeze() == y_pred.squeeze(), dim).data().item<int>()*1.0 / y.size(0);
        return accuracy;
    }
};



#endif /* BAGGING_H_ */
