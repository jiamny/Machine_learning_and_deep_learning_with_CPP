/*
 * LogisticRegression.hpp
 *
 *  Created on: May 4, 2024
 *      Author: jiamny
 */


#ifndef LOGISTICREGRESSION_HPP_
#define LOGISTICREGRESSION_HPP_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

class LogisticRegression {
public:
	LogisticRegression(torch::Tensor X) {
        /*
        :param X: Input tensor
        :keyword lr: learning rate
        :keyword epochs: number of times the model iterates over complete dataset
        :keyword weights: parameters learned during training
        :keyword bias: parameter learned during training
        */
        lr = 0.1;
        epochs = 1000;
        m = X.size(0);
        n = X.size(1);
        weights = torch::zeros({n, 1}, X.dtype());
        bias = torch::tensor({0.0}, X.dtype());
	}

	torch::Tensor loss(torch::Tensor yhat, torch::Tensor y) {
        /*
        :param yhat: Estimated y
        :return: Log loss - When y=1, it cancels out half function, remaining half is considered for loss calculation and vice-versa
        */
		torch::Tensor ls = -1.0*(1.0 / m) * torch::sum(y * torch::log(yhat) + (1.0 - y) * torch::log(1.0 - yhat));
        return ls;
	}

	std::pair<torch::Tensor, torch::Tensor> gradient(torch::Tensor X,
											torch::Tensor y_predict, torch::Tensor y) {
        /*
        :param y_predict: Estimated y
        :return: gradient is calculated to find how much change is required in parameters to reduce the loss.
        */
		torch::Tensor dw = (1.0 / m) * torch::mm(X.t(), (y_predict - y));
		torch::Tensor db = (1.0 / m) * torch::sum(y_predict - y);
        return std::make_pair(dw, db);
	}

	std::pair<torch::Tensor, torch::Tensor> run(torch::Tensor X, torch::Tensor y) {
        /*
        :param X: Input tensor
        :param y: Output tensor
        :var y_predict: Predicted tensor
        :var cost: Difference between ground truth and predicted
        :var dw, db: Weight and bias update for weight tensor and bias scalar
        :return: updated weights and bias
        */
        for(auto epoch : range(epochs, 1) ) {
            auto y_predict = Sigmoid(torch::mm(X, weights) + bias);
            auto cost = loss(y_predict, y);

            torch::Tensor dw, db;
            std::tie(dw, db) = gradient(X, y_predict, y);
            //std::cout << "dw: " << dw << '\n';
            //std::cout << "db: " << db << '\n';

            weights -= lr * dw;
            bias -= lr * db;

            if( epoch % 100 == 0 ) {
                std::cout << "Cost after iteration " << epoch << " : " << cost.data().item<double>() << '\n';
            }
        }
        return std::make_pair(weights, bias);
	}

	torch::Tensor predict(torch::Tensor X) {
        /*
        :param X: Input tensor
        :var y_predict_labels: Converts float value to int/bool true(1) or false(0)
        :return: outputs labels as 0 and 1
        */
		torch::Tensor  y_predict = Sigmoid(torch::mm(X, weights) + bias);
		torch::Tensor y_predict_labels = y_predict > 0.5;
        return y_predict_labels.to(torch::kInt);
	}

    double score(torch::Tensor X, torch::Tensor y, bool verbose = false) {
        torch::Tensor y_pred = predict(X).squeeze_();
        if(verbose) {
        	std::cout << "y: " << y.sizes() << '\n';
        	std::cout << "y_pred: " << y_pred.sizes() << '\n';
        	printVector(tensorTovector(y_pred.to(torch::kDouble)));
        }
        c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum(y.squeeze() == y_pred, dim).data().item<double>() / X.size(0);
        return accuracy;
    }

private:
       float lr = 0.0;
       int epochs = 0, m = 0, n = 0;
       torch::Tensor weights, bias;
};


#endif /* LOGISTICREGRESSION_HPP_ */
