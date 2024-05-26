/*
 * weights.h
 *
 *  Created on: May 12, 2024
 *      Author: jiamny
 */

#ifndef WEIGHTS_H_
#define WEIGHTS_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
//#include <limits>

//#include "../TempHelpFunctions.h"
//#include "../helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

std::pair<int, int> calc_fan(at::IntArrayRef weight_shape) {
    /*
    对权重矩阵计算 fan-in 和 fan-out

    参数说明：
    weight_shape：权重形状
    */
    int fan_in = 0, fan_out = 0;
    if( weight_shape.size() == 2 ) {
        fan_in = weight_shape.at(0);
        fan_out = weight_shape.at(1);
    } else if( weight_shape.size() == 3 || weight_shape.size() == 4 ) { // [3, 4]
        int in_ch = weight_shape.at(weight_shape.size() - 2);
        int out_ch = weight_shape.at(weight_shape.size() - 1);	//[-2:]
        int kernel_size = torch::prod((torch::tensor(weight_shape)).index({Slice(0, -2)})).data().item<int>();
        fan_in = in_ch * kernel_size;
        fan_out = out_ch * kernel_size;
    } else {
        std::cout << "Unrecognized weight dimension: " << weight_shape << "\n";
    }
    return std::make_pair(fan_in, fan_out);
}

class _random_uniform {
public:
   /*
    初始化网络权重 W--- 基于 Uniform(-b, b)
    参数说明：
    weight_shape：权重形状
    */
	_random_uniform() {};
	_random_uniform(double _b=1.0) {
        b = _b;
	}

    torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
        return torch::empty(weight_shape).uniform_(-b, b);
    }
private:
   double b = 1.0;
};

class _random_normal {
   /*
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)
    参数说明：
    weight_shape：权重形状
    std：权重标准差
   */
public:
	_random_normal() {};
	_random_normal(double _std) {
        std = _std;
	}

	torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
        return torch::normal(0., std, weight_shape);
	}
private:
	double std = 0.01;
};

class he_uniform {
   /*
    初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=sqrt(6/fan_in)，常用于 ReLU 激活层
    参数说明：
    weight_shape：权重形状
   */
public:
	he_uniform(){};

	torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
		int fan_in = 0, fan_out = 0;
        std::tie(fan_in, fan_out) = calc_fan(weight_shape);
        double b = std::sqrt(6. / fan_in);
        return torch::empty(weight_shape).uniform_(-b, b);
	}
};

class he_normal {
   /*
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=2/fan_in，常用于 ReLU 激活层
    参数说明：
    weight_shape：权重形状
   */
public:
	he_normal() {};

	torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
		int fan_in = 0, fan_out = 0;
        std::tie(fan_in, fan_out) = calc_fan(weight_shape);
        double std = std::sqrt(2. / fan_in);
        return torch::normal(0., std, weight_shape);
	}
};

class glorot_uniform {
   /*
    初始化网络权重 W--- 基于 Uniform(-b, b)，其中 b=gain*sqrt(6/(fan_in+fan_out))，
                        常用于 tanh 和 sigmoid 激活层
    参数说明：
    weight_shape：权重形状
   */
public:
	glorot_uniform() {};
	glorot_uniform(double _gain=1.0) {
        gain = _gain;
	}

	torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
		int fan_in = 0, fan_out = 0;
        std::tie(fan_in, fan_out) = calc_fan(weight_shape);
        double b = gain * std::sqrt(6. / (fan_in + fan_out));
        return torch::empty(weight_shape).uniform_(-b, b);
	}
private:
	double gain = 1.0;
};

class glorot_normal {
   /*
    初始化网络权重 W--- 基于 TruncatedNormal(0, std)，其中 std=gain^2*2/(fan_in+fan_out)，
                        常用于 tanh 和 sigmoid 激活层
    参数说明：
    weight_shape：权重形状
   */
	glorot_normal() {};
	glorot_normal(double _gain=1.0) {
        gain = _gain;
	}

	torch::Tensor getIntiWeight(at::IntArrayRef weight_shape) {
		int fan_in = 0, fan_out = 0;
        std::tie(fan_in, fan_out) = calc_fan(weight_shape);
        double std = gain * std::sqrt(2. / (fan_in + fan_out));
        return torch::normal(0, std, weight_shape);
	}
private:
	double gain = 1.0;
};

#endif /* WEIGHTS_H_ */
