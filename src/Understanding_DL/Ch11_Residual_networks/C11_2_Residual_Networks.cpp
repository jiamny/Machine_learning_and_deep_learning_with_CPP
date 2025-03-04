/*
 * C11_2_Residual_Networks.cpp
 *
 *  Created on: Feb 19, 2025
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

/*
# We will adapt this model to have residual connections around the linear layers
# This is the same model we used in practical 8.1, but we can't use the sequential
# class for residual networks (which aren't strictly sequential).  Hence, I've rewritten
# it as a model that inherits from a base class
*/

class ResidualNetwork : public torch::nn::Module {
public:
	ResidualNetwork(int input_size, int output_size, int hidden_size=100) {
		linear1 = torch::nn::Linear(torch::nn::LinearOptions(input_size, hidden_size));
    	linear2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
    	linear3 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
    	linear4 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_size));
    	register_module("linear1", linear1);
    	register_module("linear2", linear2);
    	register_module("linear3", linear3);
    	register_module("linear4", linear4);
    	printf("Initialized MLPBase model with %ld parameters\n", count_params());
	}

    long count_params() {
    	long cnt = 0;
    	for(auto& p : this->parameters())
    		cnt += p.view({-1}).size(0);
    	return cnt;
    }

    /*
	# Add residual connections to this model
	# The order of operations within each block should similar to figure 11.5b
	# ie., linear1 first, ReLU+linear2 in first residual block, ReLU+linear3 in second residual block), linear4 at end
	# Replace this function
	*/
    torch::Tensor forward(torch::Tensor x) {
    	torch::Tensor f = linear1->forward(x);
    	torch::Tensor res1 = f + linear2(f.relu());
    	torch::Tensor res2 = res1 + linear3->forward(res1.relu());

    	return linear4->forward(res2);
    }
private:
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr}, linear4{nullptr};
};

// Initialization of weights
void weights_init(ResidualNetwork& model){
	for(auto& module : model.modules((/*include_self=*/false))) {
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
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Load data sets\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor X_train, y_train;
	std::tie(X_train, y_train) = load_mnist1d();
	X_train = X_train.to(torch::kFloat32).t();
	y_train = y_train.to(torch::kInt64);

	torch::Tensor X_test, y_test;
	std::tie(X_test, y_test) = load_mnist1d(false);
	X_test = X_test.to(torch::kFloat32).t();
	y_test = y_test.to(torch::kInt64);

	std::cout << "Examples in training set: " << y_train.size(0) << '\n'
			  << "Examples in test set:" << y_test.size(0)  << '\n'
			  << "Length of each example: " << X_train.size(1) << '\n';
	// Print out sizes
	printf("Train data: %d examples (columns), each of which has %d dimensions (rows)\n",
			int(X_train.size(1)), int(X_train.size(0)));
	printf("Validation data: %d examples (columns), each of which has %d dimensions (rows)\n",
			int(X_test.size(1)), int(X_test.size(0)));

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Create and run the ResidualNetwork model\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// The inputs correspond to the 40 offsets in the MNIST1D template.
	int D_i = 40;
	// The outputs correspond to the 10 digits
	int D_o = 10;

	// #Define the model
	ResidualNetwork model(D_i, D_o);

	torch::Tensor x = torch::randn({100, 1, 40});
	std::cout << model.forward(x).sizes() << '\n';

	weights_init(model);
	model.to(device);


	// choose cross entropy loss function (equation 5.24)
	auto loss_function = torch::nn::CrossEntropyLoss();

	// construct SGD optimizer and initialize learning rate and momentum
	auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.05).momentum(0.9));

	// object that decreases learning rate by half every 20 epochs
	auto scheduler = torch::optim::StepLR(optimizer, 20, 0.5);

	// loop over the dataset n_epoch times
	int n_epoch = 100;

	// store the loss and the % correct at each epoch
	std::vector<double> losses_train, errors_train, losses_test, errors_test, xx;

	int batch_size = 100;
	auto dataset = LRdataset(X_train.t(), y_train).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset), batch_size);

	auto dataset_ts = LRdataset(X_test.t(), y_test).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset_ts), batch_size);


	for(auto& epoch : range(n_epoch, 1)) {
		double tr_error = 0., ts_error =0.;
		double tr_ls = 0., ts_ls = 0.;
		int num_smaples = 0;
		int correct = 0;
		model.train();

		// loop over batches
		for (auto &batch : *data_loader) {
		    // retrieve inputs and labels for this batch
			auto x_batch = batch.data.to(device);;
			auto y_batch = batch.target.flatten().to(device);
			x_batch = x_batch.unsqueeze(1);

		    // zero the parameter gradients
		    optimizer.zero_grad();

		    //forward pass -- calculate model output
		    auto pred = model.forward(x_batch).squeeze(1);

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

		model.eval();
		correct = 0;
		num_smaples = 0;
		for (auto &batch : *test_loader) {
			// retrieve inputs and labels for this batch
			auto x_batch = batch.data.to(device);
			auto y_batch = batch.target.flatten().to(device);
			x_batch = x_batch.unsqueeze(1);
			auto pred = model.forward(x_batch).squeeze(1);

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

		printf("Epoch: %5d, train loss: %.6f, train error %3.4f,  test loss: %.6f, test error: %3.4f\n",
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
	matplot::plot(fx, xx, losses_train,"r-")->line_width(2).display_name("losses train");
	matplot::plot(fx, xx, losses_test,"b-")->line_width(2).display_name("losses test");
	matplot::xlabel(fx, "epoch");
	matplot::ylabel(fx, "loss");
	matplot::legend(fx, {});

	auto fx2 = F->nexttile();
	matplot::hold(fx2, true);
	matplot::plot(fx2, xx, errors_train,"r-")->line_width(2).display_name("error train");
	matplot::plot(fx2, xx, errors_test,"b-")->line_width(2).display_name("error test");
	matplot::ylabel(fx2, "error");
	matplot::xlabel(fx2, "epoch");
	matplot::legend(fx2, {});
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}

