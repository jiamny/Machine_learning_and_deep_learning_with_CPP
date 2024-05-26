/*
 * XOR.cpp
 *
 *  Created on: May 24, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

struct _XORImpl : public torch::nn::Module {
	_XORImpl() {
        fc1 = torch::nn::Linear(torch::nn::LinearOptions(2, 3));   // 隐藏层 3个神经元
        fc2 = torch::nn::Linear(torch::nn::LinearOptions(3, 4));   // 隐藏层 4个神经元
        fc3 = torch::nn::Linear(torch::nn::LinearOptions(4, 1));   // 输出层 1个神经元
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);

        fc1->weight.data().normal_(0.0, 1.0);
        fc1->bias.data().fill_(0.);
        fc2->weight.data().normal_(0.0, 1.0);
        fc2->bias.data().fill_(0.);
        fc3->weight.data().normal_(0.0, 1.0);
        fc3->bias.data().fill_(0.);
	}

    torch::Tensor forward(torch::Tensor x) {
    	x = torch::sigmoid(fc1->forward(x));
    	x = torch::sigmoid(fc2->forward(x));
        return torch::sigmoid(fc3->forward(x));;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
TORCH_MODULE(_XOR);

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::Tensor x = torch::tensor({{1., 0.},
									 {0., 1.},
									 {1., 1.},
									 {0., 0.}}).to(torch::kFloat32).to(device);
	torch::Tensor y = torch::tensor({{1.}, {1.}, {0.}, {0.}}).to(torch::kFloat32).to(device);
	std::cout << x.sizes() << '\n';

	auto net = _XOR();
	net->to(device);
	net->train();

	// 定义loss function
	auto criterion = torch::nn::BCELoss(); // MSE
	// 定义优化器
	auto optimizer = torch::optim::SGD(net->parameters(), torch::optim::SGDOptions(0.1).momentum(0.9)); // SGD

	// 训练
	torch::AutoGradMode enable_grad(true);
	for(auto& epoch : range(3000, 0)) {
	    auto out = net->forward(x);
	    auto loss = criterion(out, y);
	    if( (epoch + 1) % 200 == 0 ) {
	    	std::cout << "epoch: " << (epoch+1) << ", loss: " << loss.data().item<float>()
	    			  << ", out:\n" << out << '\n';
	    }
	    optimizer.zero_grad();  // 清零梯度缓存区
	    loss.backward();
	    optimizer.step();  // 更新
	}

	net->eval();
	torch::NoGradGuard no_grad;
	torch::Tensor out = net->forward(x);
	std::cout << "out:\n" << out << '\n';
	std::cout << "Done!\n";
	return 0;
}
