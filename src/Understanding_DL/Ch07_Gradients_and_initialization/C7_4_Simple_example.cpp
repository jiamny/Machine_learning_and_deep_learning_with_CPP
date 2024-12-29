/*
 * C7_4_Simple_example.cpp
 *
 *  Created on: Dec 20, 2024
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

#include <matplot/matplot.h>
using namespace matplot;

// He initialization of weights
void weights_init(torch::nn::Sequential& modules){
	for(auto& module : modules->modules((/*include_self=*/false))) {
		if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
		    torch::nn::init::kaiming_normal_(
		            M->weight,
		            /*a=*/0,
		            torch::kFanOut,
		            torch::kReLU);
		    M->bias.fill_(0.0);
		}
	}
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Training two-layer network on random data\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// define input size, hidden layer size, output size
	int D_i = 10, D_k = 40, D_o = 5;

	// create model with two hidden layers
	torch::nn::Sequential model = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(D_i, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_o)));

	weights_init(model);
	model->to(device);

	// choose least squares loss function
	auto criterion = torch::nn::MSELoss();
	// construct SGD optimizer and initialize learning rate and momentum
	torch::optim::SGD optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.1).momentum(0.9));

	// object that decreases learning rate by half every 10 epochs
	//scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
	auto scheduler = torch::optim::StepLR(optimizer, 10, 0.5);

	// create 100 random data points and store in data loader class
	torch::Tensor x = torch::randn({100, D_i});
	torch::Tensor y = torch::randn({100, D_o});

	int batch_size = 10;
	auto dataset = LRdataset(x, y).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset), batch_size);

	std::vector<double> xx, yy;
	// # loop over the dataset 100 times
	for(auto& epoch : range(100, 0)) {
		float epoch_loss = 0.0;
		int num_batch = 0;
		// loop over batches
		for (auto &batch : *data_loader) {
			auto x_batch = batch.data.to(device);
			auto y_batch = batch.target.to(device);

			// zero the parameter gradients
			optimizer.zero_grad();

			// forward pass
			auto pred = model->forward(x_batch);
			auto loss = criterion(pred, y_batch);

			// backward pass
			loss.backward();

			// SGD update
			optimizer.step();

			// update statistics
			epoch_loss += loss.cpu().data().item<float>();
			num_batch += 1;
		}
		// print error
		printf("Epoch: %5d, loss: %.3f\n", epoch+1, epoch_loss);
		xx.push_back(epoch * 1.0 + 1.0);
		yy.push_back(epoch_loss/num_batch);
		// tell scheduler to consider updating learning rate
		scheduler.step();
	}

	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::plot(fx, xx, yy,"-")->line_width(3);
	matplot::xlabel("epoch");
	matplot::ylabel("loss");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}



