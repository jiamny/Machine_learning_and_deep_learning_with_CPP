/*
 * activation.hpp
 *
 *  Created on: May 10, 2024
 *      Author: jiamny
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <limits>

using torch::indexing::Slice;
using torch::indexing::None;


class ActivationBase {
public:
	virtual ~ActivationBase() {};

	virtual torch::Tensor forward(torch::Tensor z) { return z;};

	virtual torch::Tensor gradient(torch::Tensor x) { return x;};
};

class _Sigmoid : public ActivationBase {
    //_Sigmoid(x) = 1 / (1 + e^(-x))
public:
	_Sigmoid() {};

	torch::Tensor forward(torch::Tensor z) {
    	if( z.dim() == 1)
    		z = z.reshape({1, -1});

        return 1.0 / (1.0 + torch::exp(-z));
    }

	torch::Tensor gradient(torch::Tensor x) {
        return forward(x) * (1.0 - forward(x));
    }
};

class _Softmax : public ActivationBase {
public:
	_Softmax(long int dim = 0) {dm = dim;};

	torch::Tensor forward(torch::Tensor X) {
		torch::Tensor val_max = torch::max(X);
		torch::Tensor X_exp = torch::exp(X - val_max);

		c10::OptionalArrayRef<long int> dim = {dm};
		torch::Tensor partition = torch::sum(X_exp, dim, true);
		return (X_exp / partition);
	}

	torch::Tensor  gradient(torch::Tensor  X) {
		torch::Tensor p = forward(X);
        return p * (1. - p);
	}
private:
	long int dm = 0;
};

class _ReLU : public ActivationBase {
    /*
    ReLU(x) =
            x   if x > 0
            0   otherwise
    */
public:
	_ReLU() {};

	torch::Tensor forward(torch::Tensor z) {
    	//return np.clip(z, 0, np.inf)
    	if( z.dim() == 1)
    		z = z.reshape({1, -1});

    	return torch::where(z > 0.0, z, 0.0);
    }

	torch::Tensor gradient(torch::Tensor x) {
	    if( x.dim() == 1)
	    	x = x.reshape({1, -1});

        return torch::where(x >=0.0, 1.0, 0.0);
    }
};

class _Tanh : public ActivationBase {
    //Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
public:
	_Tanh() {};

	torch::Tensor forward(torch::Tensor z) {
    	//return np.clip(z, 0, np.inf)
    	if( z.dim() == 1)
    		z = z.reshape({1, -1});
    	return 2.0 / (1.0 + torch::exp(-2.0 * z)) - 1.;
    }

	torch::Tensor gradient(torch::Tensor x) {
        return 1. - torch::pow(forward(x), 2);
    }
};

class _Affine : public ActivationBase {
public:
    //Affine(x) = slope * x + intercept
	_Affine() {};
	_Affine(double _slope, double _intercept) {
        slope = _slope;
        intercept = _intercept;
	};

	torch::Tensor forward(torch::Tensor z) {
    	//return np.clip(z, 0, np.inf)
    	if( z.dim() == 1)
    		z = z.reshape({1, -1});
        return slope * z + intercept;
    }

	torch::Tensor gradient(torch::Tensor x) {
	    if( x.dim() == 1)
	    	x = x.reshape({1, -1});

        return slope * torch::ones_like(x);
    }
private:
    double slope = 1., intercept = 0.;
};

class _LeakyReLU : public ActivationBase {
public:
    /*
    LeakyReLU(x) =
            alpha * x   if x < 0
            x           otherwise
    */
	_LeakyReLU() {};
	_LeakyReLU(double  _alpha) {
        alpha = _alpha;
	}

	torch::Tensor forward(torch::Tensor z) {
    	if( z.dim() == 1)
    		z = z.reshape({1, -1});

		return torch::where(z > 0.0, z, alpha * z);
    }

	torch::Tensor gradient(torch::Tensor x) {
	    if( x.dim() == 1)
	    	x = x.reshape({1, -1});

    	return torch::where(x > 0.0, 1.0, alpha);
    }
private:
    double alpha = 0.3;
};

class _ELU : public ActivationBase {
public:
	_ELU() {};
	_ELU(double _alpha) {
        alpha = _alpha;
	}

	torch::Tensor forward(torch::Tensor z) {
	    if( z.dim() == 1)
	    	z = z.reshape({1, -1});

        return torch::where(z >= 0.0, z, alpha * (torch::exp(z) - 1.));
	}

	torch::Tensor gradient(torch::Tensor X) {
		if( X.dim() == 1)
		    X = X.reshape({1, -1});

        return torch::where(X >= 0.0, 1.0, forward(X) + alpha);
	}
private:
	 double alpha = 0.;
};

class _SELU : public ActivationBase {
public:
	_SELU() {
        alpha = 1.6732632423543772848170429916717;
        scale = 1.0507009873554804934193349852946;
	}

	torch::Tensor forward(torch::Tensor z) {
	    if( z.dim() == 1)
	    	z = z.reshape({1, -1});

        return scale * torch::where(z >= 0.0, z, alpha*(torch::exp(z)-1.));
	}

	torch::Tensor  gradient(torch::Tensor x) {
		if( x.dim() == 1)
		    x = x.reshape({1, -1});

        return scale * torch::where(x >= 0.0, 1.0, alpha * torch::exp(x));
	}

private:
    double alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946;
};

class _SoftPlus : public ActivationBase {
public:
	_SoftPlus() {};

	torch::Tensor forward(torch::Tensor z) {
	    if( z.dim() == 1)
	    	z = z.reshape({1, -1});

	    return torch::log(1. + torch::exp(z));
	}

	torch::Tensor gradient(torch::Tensor  x) {
	    if( x.dim() == 1)
	    	x = x.reshape({1, -1});

		return 1. / (1. + torch::exp(-x));
	}
};

class _Exponential : public ActivationBase {
public:
		 // Exponential(x) = e^x
	_Exponential() {};

	torch::Tensor forward(torch::Tensor z) {
	    if( z.dim() == 1)
	    	z = z.reshape({1, -1});

        return torch::exp(z);
	}

	torch::Tensor gradient(torch::Tensor  x) {
        return forward(x);
	}
};

class _HardSigmoid : public ActivationBase {
    /*
    HardSigmoid(x) =
            0               if x < -2.5
            0.2 * x + 0.5   if -2.5 <= x <= 2.5.
            1               if x > 2.5
    */
public:
	_HardSigmoid() {};

	torch::Tensor forward(torch::Tensor z) {
	    if( z.dim() == 1)
	    	z = z.reshape({1, -1});

	    const std::optional<c10::Scalar> min = {0.0};
	    const std::optional<c10::Scalar> max = {1.0};
        return torch::clip((0.2 * z) + 0.5, min, max);
	}

	torch::Tensor gradient(torch::Tensor  x) {
	    if( x.dim() == 1)
	    	x = x.reshape({1, -1});

        return torch::where(((x >= -2.5) & (x <= 2.5)), 0.2, 0);
	}
};

#endif /* ACTIVATION_H_ */
