/*
 * loss.h
 *
 *  Created on: May 24, 2024
 *      Author: jiamny
 */

#ifndef LOSS_H_
#define LOSS_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

using torch::indexing::Slice;
using torch::indexing::None;

class _MeanSquareLoss {
public:
	_MeanSquareLoss() {}

	torch::Tensor loss(torch::Tensor y, torch::Tensor y_pred) {
		c10::OptionalArrayRef<long int> dim = {1};
        return torch::sum(torch::pow((y - y_pred), 2), dim) / y.size(0);
	}

	torch::Tensor gradient(torch::Tensor y, torch::Tensor y_pred) {
        return -1.0*(y - y_pred);
	}

	torch::Tensor hess(torch::Tensor y, torch::Tensor y_pred) {
	    return torch::tensor({1});
	}
};

class _CrossEntropy {
public:
	_CrossEntropy(){};
	~_CrossEntropy() {};

    torch::Tensor loss(torch::Tensor y, torch::Tensor p) {
    	const std::optional<c10::Scalar> min = {1e-15};
    	const std::optional<c10::Scalar> max = {1-1e-15};
        p = torch::clip(p, min, max);
        return -1.* y * torch::log(p) - (1. -y) * torch::log(1. - p);
    }

    torch::Tensor accuracy_score(torch::Tensor y, torch::Tensor p) {
    	std::optional<long int> dim = {1};
        return accuracy_score(torch::argmax(y, dim), torch::argmax(p, dim));
    }

    torch::Tensor gradient(torch::Tensor y, torch::Tensor p) {
    	const std::optional<c10::Scalar> min = {1e-15};
    	const std::optional<c10::Scalar> max = {1. - 1e-15};
        p = torch::clip(p, min, max);
        return -1. * (y / p) + (1. - y) / (1. - p);
    }

	torch::Tensor hess(torch::Tensor y, torch::Tensor y_pred) {
	    return y_pred * (1 - y_pred);
	}
};

class _MeanAbsoluteLoss {
public:
	_MeanAbsoluteLoss() {};

	torch::Tensor loss(torch::Tensor y, torch::Tensor y_pred) {
		c10::OptionalArrayRef<long int> dim = {1};
        return torch::sum(torch::abs(y - y_pred), dim) / y.size(0);
    }

    torch::Tensor gradient(torch::Tensor y, torch::Tensor y_pred) {
        return -1.0*(y - y_pred);
    }
};

class _HuberLoss {
public:
	_HuberLoss() {}

	torch::Tensor loss(torch::Tensor y, torch::Tensor y_pred, torch::Tensor delta) {
        if( torch::abs(y - y_pred).le(delta).data().item<bool>() ) // <= delta
            return 0.5 * torch::pow((y - y_pred), 2);
        else
            return (delta * torch::abs(y - y_pred)) - (0.5 * torch::pow(delta, 2));
	}
};

#endif /* LOSS_H_ */
