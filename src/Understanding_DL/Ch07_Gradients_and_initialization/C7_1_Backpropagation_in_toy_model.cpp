/*
 * C7_1_Backpropagation_in_toy_model.cpp
 *
 *  Created on: Dec 16, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

torch::Tensor fn(torch::Tensor x, float beta0, float beta1, float beta2, float beta3,
					float omega0, float omega1, float omega2, float omega3) {
  return beta3+omega3 * torch::cos(beta2 + omega2 * torch::exp(beta1 + omega1 * torch::sin(beta0 + omega0 * x)));
}

torch::Tensor loss(torch::Tensor x, torch::Tensor y, float beta0, float beta1, float beta2, float beta3,
					float omega0, float omega1, float omega2, float omega3) {
	torch::Tensor diff = fn(x, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3) - y;
  return diff * diff;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Consider a network with three hidden layers h1 , h2 , and h3:\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	float beta0 = 1.0, beta1 = 2.0, beta2 = -3.0, beta3 = 0.4,
	omega0 = 0.1, omega1 = -0.4, omega2 = 2.0, omega3 = 3.0;

	torch::Tensor x = torch::tensor({2.3}), y = torch::tensor({2.0});
	float l_i_func = loss(x, y, beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3).data().item<float>();
	std::cout << "l_i = " << l_i_func << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Computing derivatives by hand:\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor dldbeta3_func = 2 * (beta3 +omega3 * torch::cos(beta2 +
						omega2 * torch::exp(beta1+omega1 * torch::sin(beta0+omega0 * x)))-y);
	torch::Tensor dldomega0_func = -2 *(beta3 +omega3 * torch::cos(beta2 +
						omega2 * torch::exp(beta1+omega1 * torch::sin(beta0+omega0 * x)))-y) *
						omega1 * omega2 * omega3 * x * torch::cos(beta0 + omega0 * x) *
						torch::exp(beta1 +omega1 * torch::sin(beta0 + omega0 * x)) *
						torch::sin(beta2 + omega2 * torch::exp(beta1+ omega1* torch::sin(beta0+omega0 * x)));


	torch::Tensor dldomega0_fd = (loss(x,y,beta0,beta1,beta2,beta3,omega0+0.00001,omega1,omega2,omega3) -
					loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3))/0.00001;

	printf("dydomega0: Function value = %3.3f, Finite difference value = %3.3f\n",
			dldomega0_func.data().item<float>(), dldomega0_fd.data().item<float>());

	float f0 = beta0 + omega0 * x.data().item<float>();
	float h1 = std::sin(f0);
	float f1 = beta1 + omega1 * h1;
	float h2 = std::exp(f1);
	float f2 = beta2 + omega2 * h2;
	float h3 = std::cos(f2);
	float f3 = beta3 + omega3 * h3;
	float l_i = std::pow(f3 - y.data().item<float>(), 2);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute all the f_k and h_k terms:\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Let's check we got that right:
	printf("f0: true value = %3.3f, your value = %3.3f\n", 1.230, f0);
	printf("h1: true value = %3.3f, your value = %3.3f\n", 0.942, h1);
	printf("f1: true value = %3.3f, your value = %3.3f\n", 1.623, f1);
	printf("h2: true value = %3.3f, your value = %3.3f\n", 5.068, h2);
	printf("f2: true value = %3.3f, your value = %3.3f\n", 7.137, f2);
	printf("h3: true value = %3.3f, your value = %3.3f\n", 0.657, h3);
	printf("f3: true value = %3.3f, your value = %3.3f\n", 2.372, f3);
	printf("l_i original = %3.3f, l_i from forward pass = %3.3f\n", l_i_func, l_i);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute the derivatives of the output with respect:\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// to the intermediate computations h_k and f_k (i.e, run the backward pass)

	float dldf3 = 2* (f3 - y.data().item<float>());
	float dldh3 = omega3 * dldf3;
	// Replace the code below
	float dldf2 = -std::sin(f2) * dldh3;
	float dldh2 = omega2 * dldf2;
	float dldf1 = std::exp(f1) * dldh2;
	float dldh1 = omega1 * dldf1;
	float dldf0 = std::cos(f0) * dldh1;

	// Let's check we got that right
	printf("dldf3: true value = %3.3f, your value = %3.3f\n", 0.745, dldf3);
	printf("dldh3: true value = %3.3f, your value = %3.3f\n", 2.234, dldh3);
	printf("dldf2: true value = %3.3f, your value = %3.3f\n", -1.683, dldf2);
	printf("dldh2: true value = %3.3f, your value = %3.3f\n", -3.366, dldh2);
	printf("dldf1: true value = %3.3f, your value = %3.3f\n", -17.060, dldf1);
	printf("dldh1: true value = %3.3f, your value = %3.3f\n", 6.824, dldh1);
	printf("dldf0: true value = %3.3f, your value = %3.3f\n", 2.281, dldf0);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Calculate the final derivatives with respect to the beta and omega terms:\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	float dldbeta3 = 1 * dldf3;
	float dldomega3 = h3 * dldf3;
	float dldbeta2 = 1 * dldf2;
	float dldomega2 = h2 * dldf2;
	float dldbeta1 = 1 * dldf1;
	float dldomega1 = h1 * dldf1;
	float dldbeta0 = 1 * dldf0;
	float dldomega0 = l_i * dldf0;

	// Let's check we got them right
	printf("dldbeta3: Your value = %3.3f, True value = %3.3f\n", dldbeta3, 0.745);
	printf("dldomega3: Your value = %3.3f, True value = %3.3f\n", dldomega3, 0.489);
	printf("dldbeta2: Your value = %3.3f, True value = %3.3f\n", dldbeta2, -1.683);
	printf("dldomega2: Your value = %3.3f, True value = %3.3f\n", dldomega2, -8.530);
	printf("dldbeta1: Your value = %3.3f, True value = %3.3f\n", dldbeta1, -17.060);
	printf("dldomega1: Your value = %3.3f, True value = %3.3f\n", dldomega1, -16.079);
	printf("dldbeta0: Your value = %3.3f, True value = %3.3f\n", dldbeta0, 2.281);
	printf("dldomega0: Your value = %3.3f, Function value = %3.3f, Finite difference value = %3.3f\n",
			dldomega0, dldomega0_func.data().item<float>(), dldomega0_fd.data().item<float>());

	std::cout << "Done!\n";
	return 0;
}


