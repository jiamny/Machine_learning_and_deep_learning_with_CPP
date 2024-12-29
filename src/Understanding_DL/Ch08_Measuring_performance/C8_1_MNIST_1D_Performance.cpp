/*
 * C8_1_MNIST_1D_Performance.cpp
 *
 *  Created on: Dec 19, 2024
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
	std::cout << "// Load data sets\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor X_train, y_train;
	std::tie(X_train, y_train) = load_mnist1d();
	X_train = X_train.to(torch::kFloat32);
	y_train = y_train.to(torch::kInt64);

	torch::Tensor X_test, y_test;
	std::tie(X_test, y_test) = load_mnist1d(false);
	X_test = X_test.to(torch::kFloat32);
	y_test = y_test.to(torch::kInt64);

	std::cout << X_train.index({Slice(0,10), Slice()}) << '\n' << y_test.index({Slice(0,10)})  << '\n';

	int D_i = 40;    // Input dimensions
	int D_k = 100;   // Hidden dimensions
	int D_o = 10;    // Output dimensions

	// Define a model with two hidden layers of size 100
	// And ReLU activations between them
	// Replace this line (see Figure 7.8 of book for help):

	torch::nn::Sequential model = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(D_i, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_o)));

	weights_init(model);
	model->to(device);

	// choose cross entropy loss function (equation 5.24)
	auto loss_function = torch::nn::CrossEntropyLoss();

	// construct SGD optimizer and initialize learning rate and momentum
	auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.05).momentum(0.9));

	// object that decreases learning rate by half every 10 epochs
	auto scheduler = torch::optim::StepLR(optimizer, 10, 0.5);

	// loop over the dataset n_epoch times
	int n_epoch = 50;

	// store the loss and the % correct at each epoch
	std::vector<double> losses_train, errors_train, losses_test, errors_test, xx;

	int batch_size = 100;
	auto dataset = LRdataset(X_train, y_train).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset), batch_size);

	auto dataset_ts = LRdataset(X_test, y_test).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset_ts), batch_size);

/*
	auto batch = *data_loader->begin();
	auto data  = batch.data.to(device);
	auto y  = batch.target.flatten().to(device);
	std::cout << "y: " << y.dtype() << " " << y.sizes() << std::endl;

	auto y_hat = model->forward(data);
	std::cout << "y_hat: " << y_hat << std::endl;
	auto l = loss_function(y_hat, y);
	std::cout << l.item<float>() * data.size(0) << std::endl;
*/

	for(auto& epoch : range(n_epoch, 1)) {
		double tr_error = 0., ts_error =0.;
		double tr_ls = 0., ts_ls = 0.;
		int num_smaples = 0;
		int correct = 0;
		model->train();
		// loop over batches
		for (auto &batch : *data_loader) {
		    // retrieve inputs and labels for this batch
			auto x_batch = batch.data.to(device);;
			auto y_batch = batch.target.flatten().to(device);;

		    // zero the parameter gradients
		    optimizer.zero_grad();

		    //forward pass -- calculate model output
		    auto pred = model->forward(x_batch);

		    // compute the loss
		    auto loss = loss_function(pred, y_batch);

		    // backward pass
		    loss.backward();

		    // SGD update
		    optimizer.step();

		    tr_ls += loss.data().item<float>();
		    torch::Tensor _, predicted_train_class;
			std::tie(_, predicted_train_class) = torch::max(pred.data(), 1);
		    correct += torch::sum((predicted_train_class == y_batch).to(torch::kInt)).data().item<int>();
		    num_smaples += x_batch.size(0);
		}
		xx.push_back(epoch * 1.0);
		losses_train.push_back(tr_ls);
		tr_error = 100. - (correct*1.0/num_smaples)*100;
		errors_train.push_back(tr_error);

		model->eval();
		correct = 0;
		num_smaples = 0;
		for (auto &batch : *test_loader) {
			// retrieve inputs and labels for this batch
			auto x_batch = batch.data.to(device);
			auto y_batch = batch.target.flatten().to(device);
			auto pred = model->forward(x_batch);

			// compute the loss
			auto loss = loss_function(pred, y_batch);
			ts_ls += loss.data().item<float>();
			torch::Tensor _, predicted_train_class;
			std::tie(_, predicted_train_class) = torch::max(pred.data(), 1);
			correct += torch::sum((predicted_train_class == y_batch).to(torch::kInt)).data().item<int>();
			num_smaples += x_batch.size(0);
		}

		losses_test.push_back(ts_ls);
		ts_error = 100. - (correct*1.0/num_smaples)*100;
		errors_test.push_back(ts_error);

		printf("Epoch: %5d, train loss: %.6f, train error %3.2f,  test loss: %.6f, test error: %3.2f\n",
				epoch, tr_ls, tr_error, ts_ls, ts_error);

		// tell scheduler to consider updating learning rate
		scheduler.step();
	}

	auto F = figure(true);
	F->size(1200, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 2);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);
	matplot::plot(fx, xx, losses_train,"r-:")->line_width(3).display_name("losses train");
	matplot::plot(fx, xx, losses_test,"m--")->line_width(3).display_name("losses test");
	matplot::xlabel(fx, "epoch");
	matplot::ylabel(fx, "loss");
	matplot::legend(fx, {});

	auto fx2 = F->nexttile();
	matplot::hold(fx2, true);
	matplot::plot(fx2, xx, errors_train,"r-:")->line_width(3).display_name("error train");
	matplot::plot(fx2, xx, errors_test,"m--")->line_width(3).display_name("error test");
	matplot::ylabel(fx2, "error");
	matplot::xlabel(fx2, "epoch");
	matplot::legend(fx2, {});
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




