/*
 * SupportVectorMachine.hpp
 *
 *  Created on: May 5, 2024
 *      Author: jiamny
 */

#ifndef SUPPORTVECTORMACHINE_HPP_
#define SUPPORTVECTORMACHINE_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

class SVM {
public:

	SVM(torch::Tensor X, torch::Tensor y, double _C = 1.0) {
        total_samples = X.size(0);
        features_count = X.size(1);
        n_classes = std::get<0>(torch::_unique(y)).size(0);
        learning_rate = 0.001;
        C = _C;
	}

	torch::Tensor loss(torch::Tensor X, torch::Tensor W, torch::Tensor y) {
        /*
        C parameter tells the SVM optimization how much you want to avoid misclassifying each training
        example. For large values of C, the optimization will choose a smaller-margin hyperplane if that
        hyperplane does a better job of getting all the training points classified correctly. Conversely,
        a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
        even if that hyperplane misclassifies more points. For very tiny values of C, you should get
        misclassified examples, often even if your training data is linearly separable.

        :param X:
        :param W:
        :param y:
        :return:
        */
        int64_t num_samples = X.size(0);
        torch::Tensor distances = 1.0 - y * (torch::mm(X, W.t()));
        distances.masked_fill_(distances < 0.0,  0.0);
        torch::Tensor hinge_loss = C * (torch::sum(distances)*1.0 / num_samples).to(torch::kInt);
        torch::Tensor cost = 0.5 * torch::mm(W, W.t()) + hinge_loss;
        return cost;
	}

	torch::Tensor gradient_update(torch::Tensor W, torch::Tensor X, torch::Tensor y) {
        /*
        :param W: Weight Matrix
        :param X: Input Tensor
        :param y: Ground truth tensor
        :return: change in weight
        */

		torch::Tensor distance = 1 - (y * torch::mm(X, W.t()));
		torch::Tensor dw = torch::zeros({1, X.size(1)}, torch::kDouble);

        for(int64_t idx = 0; idx < distance.size(0); idx++) {
        	const double dist = distance[idx].data().item<double>();
        	torch::Tensor di;
            if( std::max(0.0, dist) == 0.0) {
                di = W;
            } else {
                di = W - (C * y[idx] * X.select(0, idx));
            }
            dw += di;
        }

        dw = dw*1.0 / y.size(0);
        return dw;
	}

	torch::Tensor fit(torch::Tensor X, torch::Tensor y, int max_epochs) {
        /*
        :param X: Input Tensor
        :param y: Output tensor
        :param max_epochs: Number of epochs the complete dataset is passed through the model
        :return: learned weight of the svm model
        */
		//torch::NoGradGuard no_grad;

		torch::Tensor weight = torch::randn({1, X.size(1)}, torch::kDouble) * std::sqrt(1.0 / X.size(1));

        double cost_threshold = 0.0001;
        double previous_cost = std::numeric_limits<double>::infinity();
        int nth = 0;

        for(auto&  epoch : range(max_epochs+1, 1)) {
            //X, y = shuffle(X, y)
        	// -------------------------------------------------------------
        	// shuffle data
        	// -------------------------------------------------------------
        	torch::Tensor sidx = RangeToensorIndex(X.size(0), true);
        	X = torch::index_select(X, 0, sidx.squeeze());
        	y = torch::index_select(y, 0, sidx.squeeze());

            for(int64_t idx = 0; idx < X.size(0); idx++) {
            	torch::Tensor x = X.select(0, idx);
            	torch::Tensor weight_update = gradient_update(weight, (x.clone().detach()).unsqueeze(0), y[idx]);
                weight = weight - (learning_rate * weight_update);
            }

            if( epoch % 100 == 0 ) {
            	double cost = loss(X, weight, y).data().item<double>();
                std::cout << "Loss at epoch: " << epoch << " cost: " << to_string_with_precision(cost)
                		  << " abs(previous_cost - cost): " << std::abs(previous_cost - cost) << '\n';
                if(std::abs(previous_cost - cost) < cost_threshold * previous_cost)
                    return weight;
                previous_cost = cost;
                nth += 1;
            }
        }
        return weight;
    }

    double score(torch::Tensor X_ts, torch::Tensor y_ts, torch::Tensor weight, bool verbose = false) {
    	torch::Tensor y_pred = torch::sign(torch::mm(X_ts, weight.t())).squeeze();
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
	int64_t total_samples, features_count, n_classes;
	double C, learning_rate;
};


#endif /* SUPPORTVECTORMACHINE_HPP_ */
