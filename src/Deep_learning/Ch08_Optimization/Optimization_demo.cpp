/*
 * Optimization_demo.cpp
 *
 *  Created on: May 31, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>

#include "../../Utils/fashion.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor net(torch::Tensor X, torch::Tensor& W1, torch::Tensor& b1, torch::Tensor& W2, torch::Tensor& b2) {
	_ReLU relu;
	//auto output = torch::relu(torch::add(torch::mm(X, W1), b1));  // Here '@' stands for matrix multiplication
	auto output = relu.forward(torch::add(torch::mm(X, W1), b1));   // Here '@' stands for matrix multiplication
	output = torch::add(torch::mm(output, W2), b2);
	return output;
}

std::vector<torch::Tensor> prepare_parameters(torch::Device device) {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device).requires_grad(true);

	int64_t num_inputs = 784, num_outputs=10, num_hiddens = 256;

	torch::Tensor W1 = torch::randn({num_inputs, num_hiddens}, options);
	torch::Tensor b1 = torch::zeros(num_hiddens, options);
	torch::Tensor W2 = torch::randn({num_hiddens, num_outputs}, options);
	torch::Tensor b2 = torch::zeros(num_outputs, options);

	std::cout << "W1:\n" << W1.data().index({Slice(None, 2), Slice(None, 10)}) << std::endl;
	// create optimizer parameters
	std::vector<torch::Tensor> params = {W1, b1, W2, b2};

	return params;
}

std::vector<double> optimization_demo(torch::nn::CrossEntropyLoss criterion, torch::optim::Optimizer& trainer,
		std::vector<torch::Tensor> params, std::string optim, torch::Device device) {

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// " << optim << " optimizer\n";
	std::cout << "// --------------------------------------------------\n";

	int64_t num_epochs = 100;
	int64_t batch_size = 256;

	torch::Tensor W1 = params[0], b1 = params[1], W2 = params[2], b2 = params[3];


	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
				    		.map(torch::data::transforms::Stack<>());

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
			                .map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();
	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
											         std::move(test_dataset), batch_size);
	/*
	 * Training
	 */
	std::vector<double> t_loss;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {

		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			//auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto y = batch.target.to(device);

			// ------------------------------------------------------
			// model net
			// ------------------------------------------------------
			auto y_hat = net(x.to(torch::kDouble), W1, b1, W2, b2);

			auto loss = criterion(y_hat, y); // cross_entropy(y_hat, y);

			// Update running loss
			epoch_loss += loss.item<float>() * x.size(0);

			// Update number of correctly classified samples
			epoch_correct += accuracy( y_hat, y);

			trainer.zero_grad();
			loss.backward();
			trainer.step();

			num_train_samples += x.size(0);
		}

		double sample_mean_loss = epoch_loss*1.0 / num_train_samples;

		if((epoch+1) % 20 == 0)
			std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
					            << sample_mean_loss << '\n';

		t_loss.push_back(sample_mean_loss);
	}
	std::cout << "Testing...\n";

	torch::NoGradGuard no_grad;

	int64_t epoch_correct = 0;
	num_test_samples = 0;

	for(auto& batch : *test_loader) {
		//auto data = batch.data.view({batch.data.size(0), -1}).to(device);
		auto data = batch.data.view({batch.data.size(0), -1}).to(device);
		auto target = batch.target.to(device);

		// ------------------------------------------------------
		// model net
		// ------------------------------------------------------
		auto output = net(data.to(torch::kDouble), W1, b1, W2, b2);

		auto loss = criterion(output, target);

		epoch_correct += accuracy( output, target );

		num_test_samples += data.size(0);
	}

	auto test_accuracy = static_cast<double>(epoch_correct) / num_test_samples;

	std::cout << optim << " -- accuracy: " << test_accuracy << "\n\n";

	return t_loss;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Loss Function
	auto criterion = torch::nn::CrossEntropyLoss();
	std::string optim = "SGD";

	std::vector<torch::Tensor>  params = prepare_parameters(device);
	std::vector<torch::optim::OptimizerParamGroup> parameters;
	parameters.push_back(torch::optim::OptimizerParamGroup(params));

	float lr = 0.001;
	torch::optim::SGD trainer = torch::optim::SGD(parameters, lr);
	std::vector<double> y_sgd = optimization_demo(criterion, trainer, params, optim, device);

	optim = "Adam";
	params = prepare_parameters(device);
	parameters.clear();
	parameters.push_back(torch::optim::OptimizerParamGroup(params));

	torch::optim::Adam a_trainer = torch::optim::Adam(parameters, lr);
	std::vector<double> y_adam = optimization_demo(criterion, a_trainer, params, optim, device);

	optim = "Adagrad";
	params = prepare_parameters(device);
	parameters.clear();
	parameters.push_back(torch::optim::OptimizerParamGroup(params));
	lr = 0.01;

	torch::optim::Adagrad ad_trainer = torch::optim::Adagrad(parameters, lr);
	std::vector<double> y_adagrad = optimization_demo(criterion, ad_trainer, params, optim, device);

	optim = "RMSprop";
	params = prepare_parameters(device);
	parameters.clear();
	parameters.push_back(torch::optim::OptimizerParamGroup(params));

	torch::optim::RMSprop r_trainer = torch::optim::RMSprop(parameters,
			torch::optim::RMSpropOptions(lr).weight_decay(0.95));
	std::vector<double> y_rmsprop = optimization_demo(criterion, r_trainer, params, optim, device);

	optim = "AdamW";
	params = prepare_parameters(device);
	parameters.clear();
	parameters.push_back(torch::optim::OptimizerParamGroup(params));
	lr = 0.001;
	torch::optim::AdamW aw_trainer = torch::optim::AdamW(parameters,
			torch::optim::AdamWOptions(lr));
	std::vector<double> y_adamw = optimization_demo(criterion, aw_trainer, params, optim, device);

	std::vector<double> xx;
	for(auto& i : range(100, 0))
		xx.push_back(i *1.0);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	plot(ax, xx, y_sgd, "r;")->line_width(2).display_name("SGD");
	hold(ax, on);
	plot(ax, xx, y_adam, "b-")->line_width(2).display_name("Adam");
	plot(ax, xx, y_adagrad, "g-.")->line_width(2).display_name("AdaGrad");
	plot(ax, xx, y_rmsprop, "m--")->line_width(2).display_name("RMSprop");
	//plot(ax, xx, y_adamw, "m--")->line_width(2).display_name("AdamW");
	xlabel("epoch");
	ylabel("loss");
	legend(ax, {});
	F->draw();
	show();

	std::cout << "Done!\n";
	return 0;
}



