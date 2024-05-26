/*
 * optimizer.h
 *
 *  Created on: May 12, 2024
 *      Author: jiamny
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

//#include "../TempHelpFunctions.h"
//#include "../helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class OptimizerBase {
public:
	OptimizerBase() {};
	virtual ~OptimizerBase() {};
    /*
          参数说明：
        params：待更新参数， 如权重矩阵 W；
        params_grad：待更新参数的梯度；
        params_name：待更新参数名；
    */

	virtual torch::Tensor update(torch::Tensor params, torch::Tensor params_grad) const;
};

class _SGD : public OptimizerBase {
    // sgd 优化方法
public:
    _SGD() {};
	_SGD(double _lr) {
        lr = _lr;
	}

	torch::Tensor update(torch::Tensor params, torch::Tensor params_grad) {
	   torch::Tensor update_value = lr * params_grad;
       return (params - update_value);
   }

private:
    double lr=0.01;
};

class _Momentum : public OptimizerBase {
public:
	_Momentum() {};
	_Momentum(double _lr, double _momentum) {
    /*
        参数说明：
        lr： 学习率，float (default: 0.001)
        momentum：考虑 Momentum 时的 alpha，决定了之前的梯度贡献衰减得有多快，取值范围[0, 1]，默认0
    */
       lr = _lr;
       momentum = _momentum;
       cache = torch::empty(0);
	}

	torch::Tensor update(torch::Tensor param, torch::Tensor param_grad) {
        if( cache.numel() == 0 )
            cache = torch::zeros_like(param_grad);

        torch::Tensor p_update = momentum * cache - lr * param_grad;
        cache = p_update.clone();
        return param + p_update;
	}

private:
    double lr = 0.01;
    double momentum=0.0;
    torch::Tensor cache = torch::empty(0);

};

class StochasticGradientDescentWithMomentum : public OptimizerBase {
public:
	StochasticGradientDescentWithMomentum() {};
	StochasticGradientDescentWithMomentum(double learning_rate, double _momentum) {
        lr = learning_rate;
        momentum = _momentum;
        w_update = torch::empty(0);
	}

	torch::Tensor update(torch::Tensor w, torch::Tensor gradient_wrt_w) {
        if(w_update.numel() == 0 )
            w_update = torch::zeros(w.sizes());

        w_update = momentum * w_update + (1 - momentum) * gradient_wrt_w;
        return w - lr * w_update;
	}
private:
     double lr = 0.01;
     double momentum = 0.0;
     torch::Tensor w_update = torch::empty(0);
};

class _Adagrad : public OptimizerBase {
public:
	_Adagrad() {};
	_Adagrad(double learning_rate) {
        lr = learning_rate;
        G = torch::empty(0);
        eps = 1e-8;
	}

	torch::Tensor update(torch::Tensor w, torch::Tensor gradient_wrt_w) {
        if( G.numel() == 0 )
            G = torch::zeros(w.sizes());

        G += torch::pow(gradient_wrt_w, 2);
        return w - lr * gradient_wrt_w / torch::sqrt(G + eps);
    }
private:
    double lr = 0.01;
    torch::Tensor G = torch::empty(0);
    double eps = 1e-8;
};

class _Adadelta : public OptimizerBase {
public:
	_Adadelta() {};
	_Adadelta(double _rho, double _eps) {
        E_W_update = torch::empty(0);
        E_gradient = torch::empty(0);
        w_update = torch::empty(0);
        eps = _eps;
        decay = _rho;
	}

	torch::Tensor update(torch::Tensor w, torch::Tensor gradient_wrt_w) {
        if(w_update.numel() == 0) {
            w_update = torch::zeros(w.sizes());
            E_gradient = torch::zeros(gradient_wrt_w.sizes());
            E_W_update = torch::zeros(w.sizes());
        }

        E_gradient = decay * E_gradient + (1. - decay) * torch::pow(gradient_wrt_w, 2);
        torch::Tensor RMS_Delta_W = torch::sqrt(E_W_update + eps);
        torch::Tensor RMS_gradient = torch::sqrt(E_gradient + eps);

        torch::Tensor adaptive_lr = RMS_Delta_W / RMS_gradient;
        w_update = adaptive_lr * gradient_wrt_w;
        E_W_update = decay * E_W_update + (1. - decay) * torch::pow(w_update, 2);
        return w - w_update;
	}
private:
	double decay=0.95;
	double eps=1e-6;
	torch::Tensor E_W_update = torch::empty(0);
	torch::Tensor E_gradient = torch::empty(0);
	torch::Tensor w_update = torch::empty(0);
};

class _RMSprop : public OptimizerBase {
public:
	_RMSprop() {};
	_RMSprop(double learning_rate, double _rho) {
        lr = learning_rate;
        Eg = torch::empty(0);
        eps = 1e-8;
        rho = _rho;
	}

	virtual torch::Tensor update(torch::Tensor w, torch::Tensor gradient_wrt_w) {
        if( Eg.numel() == 0)
            Eg = torch::zeros(gradient_wrt_w.sizes());

        Eg = rho * Eg + (1. -  rho) * torch::pow(gradient_wrt_w, 2);
        return w - lr * gradient_wrt_w / torch::sqrt(Eg + eps);
	}
private:
    double lr = 0.01;
    torch::Tensor Eg = torch::empty(0);
    double eps = 1e-8;
    double rho = 0.9;
};

class _Adam : public OptimizerBase {
public:
	_Adam() {};
	_Adam(double learning_rate, double _b1, double _b2) {
        lr = learning_rate;
        eps = 1e-8;
        m = torch::empty(0);
        v = torch::empty(0);
        b1 = _b1;
        b2 = _b2;
	}

	torch::Tensor update(torch::Tensor w, torch::Tensor gradient_wrt_w) {
        if( m.numel() == 0 ) {
            m = torch::zeros(gradient_wrt_w.sizes());
            v = torch::zeros(gradient_wrt_w.sizes());
        }

        m = b1 * m + (1. - b1) * gradient_wrt_w;
        v = b2 * v + (1. - b2) * torch::pow(gradient_wrt_w, 2);

		torch::Tensor m_hat = m / (1. - b1);
		torch::Tensor v_hat = v / (1. - b2);

		torch::Tensor w_update = (lr * m_hat) / (torch::sqrt(v_hat) + eps);

        return w - w_update;
	}
private:
    double lr = 0.001;
    double eps = 1e-8;
    torch::Tensor m = torch::empty(0);
    torch::Tensor v = torch::empty(0);
    double b1 = 0.9;
    double b2 = 0.999;
};

#endif /* OPTIMIZER_H_ */
