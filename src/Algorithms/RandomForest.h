/*
 * RandomFores.h
 *
 *  Created on: May 26, 2024
 *      Author: jiamny
 */

#ifndef RANDOMFORES_H_
#define RANDOMFORES_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"
#include "Decision_tree.h"

using torch::indexing::Slice;
using torch::indexing::None;

std::vector<std::pair<torch::Tensor, torch::Tensor>> get_random_subsets(
		torch::Tensor X, torch::Tensor y, int n_subsets, bool replacements=true) {
    //从训练数据中抽取数据子集 (默认可重复抽样)"""
    int n_samples = X.size(0);

    // 将 X 和 y 拼接，并将元素随机排序
	torch::Tensor sidx = RangeTensorIndex(X.size(0), true);

	X = torch::index_select(X, 0, sidx.squeeze());
	y = torch::index_select(y, 0, sidx.squeeze());

	std::vector<std::pair<torch::Tensor, torch::Tensor>> subsets;
    // 如果抽样时不重复抽样，可以只使用 50% 的训练数据；如果抽样时可重复抽样，使用全部的训练数据，默认可重复抽样
    int subsample_size = static_cast<int>(n_samples / 2);
    if( replacements )
        subsample_size = n_samples;

    std::vector<int64_t> sample_idxes;
    for(int64_t i = 0; i < n_samples; i++)
    	sample_idxes.push_back(i);

    for(auto& _ : range(n_subsets, 0) ) {
    	std::vector<int64_t> idxes = random_choice(n_samples, sample_idxes);
    	//printVector(idxes);

    	torch::Tensor idx = std::get<0>(torch::from_blob(idxes.data(), {static_cast<int64_t>(idxes.size()), 1},
    				at::TensorOptions(torch::kLong)).clone().sort(0));

        torch::Tensor XX = torch::index_select(X, 0, idx.squeeze()).clone();
        torch::Tensor yy = torch::index_select(y, 0, idx.squeeze()).clone();
        subsets.push_back(std::make_pair(XX, yy));
    }
    return subsets;
}

class RandomForest {
    // 随机森林分类器。使用一组分类树，这些分类树使用特征的随机子集训练数据的随机子集。
public:
	int n_estimators = 0, max_depth=0, min_features = 0;
	std::vector<torch::Tensor> feature_index;
	std::vector<DecisionTree_CART> trees;

	RandomForest(int _n_estimators, int _min_features, int _max_depth=30) {
        n_estimators = _n_estimators;			// 树的数目
        min_features = _min_features;			// 每棵树的最小使用特征数
        max_depth = _max_depth;					// 树的最大深度

        // 初始化决策树
        trees.clear();
        for(auto& _ : range(n_estimators, 0)) {
            trees.push_back(
            		DecisionTree_CART(max_depth));
            feature_index.push_back(torch::empty(0));
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

            // 选择特征的随机子集
        	std::vector<int> idx;
            if( min_features <  n_features) {
            	int s_features = RandT(min_features, n_features);
            	while(idx.size() < s_features) {
            		int d = RandT(0, n_features);
            		std::vector<int>::iterator it;
            		it = std::find(idx.begin(), idx.end(), d);
            		if(it == idx.end())
            			idx.push_back(d);
            	}
            	std::sort(idx.begin(), idx.end());
            } else {
            	idx = samples;
            }

            torch::Tensor sidx = torch::from_blob(idx.data(), {static_cast<int64_t>(idx.size()), 1},
                				at::TensorOptions(torch::kInt32)).clone();
            sidx = sidx.to(torch::kLong);

            // 保存特征的索引用于预测
            feature_index[i] = sidx.clone();

            // 选择索引对应的特征
            X_subset = torch::index_select(X_subset, 1, sidx.squeeze()).clone();
            // 用特征子集和真实值训练一棵子模型 (这里的数据也是训练数据集的随机子集)
            trees[i].fit(X_subset, y_subset);
        }
    }

    torch::Tensor predict(torch::Tensor X) {
    	torch::Tensor y_preds = torch::zeros({X.size(0), static_cast<int64_t>(trees.size())}, torch::kInt32);
        // 每棵决策树都在数据上预测
        for( int i = 0; i < trees.size(); i++ ) {
            // 使用该决策树训练使用的特征
        	torch::Tensor idx = feature_index[i];
        	DecisionTree_CART tree = trees[i];

            // 基于特征做出预测
        	torch::Tensor Xs = torch::index_select(X, 1, idx.squeeze()).clone();
        	std::vector<int> prediction = tree.predict(Xs);
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

#endif /* RANDOMFORES_H_ */
