/*
 * KNearestNeighbors.hpp
 *
 *  Created on: May 6, 2024
 *      Author: jiamny
 */

#ifndef KNEARESTNEIGHBORS_KNEARESTNEIGHBORS_HPP_
#define KNEARESTNEIGHBORS_KNEARESTNEIGHBORS_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;


class KNN {
public:
	KNN(int _k, torch::Tensor X) {
        //:param k: Number of Neighbors
        k = _k;
	}

	torch::Tensor distance(torch::Tensor point_1, torch::Tensor point_2,
    		std::string method = "euclidean", int p=2) {
        if( method == "euclidean" )
            return torch::norm(point_1 - point_2, 2);
        else if(method == "manhattan")
            return torch::sum(torch::abs(point_1 - point_2));
        else if( method == "minkowski" ) {
        	torch::Tensor t = torch::pow(torch::abs(point_1.sub(point_2)), p);
            return torch::pow(torch::sum(t), 1.0/p);
        } else {
            std::cout << "Unknown similarity distance type\n";
            return torch::empty(0);
        }
	}

	torch::Tensor fit_predict(torch::Tensor X, torch::Tensor y, torch::Tensor item) {
        /*
        * Iterate through each datapoints (item/y_test) that needs to be classified
        * Find distance between all train data points and each datapoint (item/y_test)
          using euclidean distance
        * Sort the distance using argsort, it gives indices of the y_test
        * Find the majority label whose distance closest to each datapoint of y_test.
        :param X: Input tensor
        :param y: Ground truth label
        :param item: tensors to be classified
        :return: predicted labels
        */

		std::vector<torch::Tensor> y_predict;

        for(int64_t i = 0; i < item.size(0); i++ ) {
        	torch::Tensor it = item.index({i, Slice()}).clone();

            std::vector<torch::Tensor> pt_distances;
            for(int64_t j = 0; j < X.size(0); j++ ) {
            	torch::Tensor distances = distance(X.index({j, Slice()}), it);
                pt_distances.push_back(torch::tensor({distances.data().item<double>()}).clone());
            }

            torch::Tensor point_distances = torch::cat(pt_distances, 0);
            torch::Tensor k_neighbors = torch::argsort(point_distances, 0, false);
            k_neighbors = k_neighbors.index({Slice(0, k)}); //[:k]
            std::vector<double> a = tensorTovector(point_distances.to(torch::kDouble));

            torch::Tensor y_label = y.index_select(0, k_neighbors); //y[k_neighbors]
            torch::Tensor major_class = std::get<0>(torch::mode(y_label, 0)); //torch::mode(y_label, true);

            y_predict.push_back(torch::tensor({major_class.data().item<long>()}));
        }

        return torch::cat(y_predict, 0);
	}

	double accuracy_score(torch::Tensor y_pred, torch::Tensor y_ts) {
        c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum(y_ts.squeeze() == y_pred.squeeze(), dim).data().item<double>() / y_ts.size(0);
        return accuracy;
	}

private:
	int k;
};


#endif /* KNEARESTNEIGHBORS_KNEARESTNEIGHBORS_HPP_ */
