/*
 * C11_3_Batch_normalization.cpp
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

// Initialization of weights
void weights_init(torch::nn::Module& model){
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

void print_variance(std::string name, torch::Tensor data) {
    // First dimension(rows) is batch elements
    // Second dimension(columns) is neurons.
	torch::Tensor t_data = data.detach();
    // Compute variance across neurons and average these variances over members of the batch
	torch::Tensor neuron_variance = torch::mean(torch::var(t_data, 0));
    // Print out the name and the variance
    printf("%40s variance = %2.6f\n", name.c_str(), neuron_variance.data().item<double>());
}

// This is a simple residual model with 5 residual branches in a row
class ResidualNetwork : public torch::nn::Module {
public:
	ResidualNetwork(int input_size, int output_size, int hidden_size=100) {
		linear1 = torch::nn::Linear(torch::nn::LinearOptions(input_size, hidden_size));
		linear2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear3 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear4 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear5 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear6 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear7 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_size));
    	register_module("linear1", linear1);
    	register_module("linear2", linear2);
    	register_module("linear3", linear3);
    	register_module("linear4", linear4);
    	register_module("linear5", linear5);
    	register_module("linear6", linear6);
    	register_module("linear7", linear7);
    	printf("Initialized MLPBase model with %ld parameters\n", count_params());
	}

    long count_params() {
    	long cnt = 0;
    	for(auto& p : this->parameters())
    		cnt += p.view({-1}).size(0);
    	return cnt;
    }

    torch::Tensor forward(torch::Tensor x) {
    	print_variance("Input",x);
      	torch::Tensor f = linear1->forward(x);
      	print_variance("First preactivation",f);
      	torch::Tensor res1 = f + linear2->forward(f.relu());
    	print_variance("After first residual connection",res1);
    	torch::Tensor res2 = res1 + linear3->forward(res1.relu());
    	print_variance("After second residual connection",res2);
    	torch::Tensor res3 = res2 + linear4->forward(res2.relu());
    	print_variance("After third residual connection",res3);
    	torch::Tensor res4 = res3 + linear5->forward(res3.relu());
    	print_variance("After fourth residual connection",res4);
    	torch::Tensor res5 = res4 + linear6->forward(res4.relu());
    	print_variance("After fifth residual connection",res5);
    	return linear7->forward(res5);
    }

private:
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr}, linear4{nullptr},
	linear5{nullptr}, linear6{nullptr}, linear7{nullptr};
};

// before the contents of each residual link as in figure 11.6c in the book
// Use the torch function nn.BatchNorm1d
class ResidualNetworkWithBatchNorm : public torch::nn::Module {
public:
	ResidualNetworkWithBatchNorm(int input_size, int output_size, int hidden_size=100) {
		linear1 = torch::nn::Linear(torch::nn::LinearOptions(input_size, hidden_size));
		linear2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear3 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear4 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear5 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear6 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size));
		linear7 = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_size));
		bn1 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));
		bn2 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));
		bn3 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));
		bn4 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));
		bn5 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));
		bn6 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1));

    	register_module("linear1", linear1);
    	register_module("linear2", linear2);
    	register_module("linear3", linear3);
    	register_module("linear4", linear4);
    	register_module("linear5", linear5);
    	register_module("linear6", linear6);
    	register_module("linear7", linear7);
    	register_module("bn1", bn1);
    	register_module("bn2", bn2);
    	register_module("bn3", bn3);
    	register_module("bn4", bn4);
    	register_module("bn5", bn5);
    	register_module("bn6", bn6);
    	printf("Initialized MLPBase model with %ld parameters\n", count_params());
	}

    long count_params() {
    	long cnt = 0;
    	for(auto& p : this->parameters())
    		cnt += p.view({-1}).size(0);
    	return cnt;
    }

    torch::Tensor forward(torch::Tensor x) {
    	print_variance("Input",x);
      	torch::Tensor f = linear1->forward(bn1->forward(x).relu());
      	print_variance("First preactivation",f);

      	torch::Tensor res1 = f + linear2->forward(bn2->forward(f).relu());
    	print_variance("After first residual connection",res1);

    	torch::Tensor res2 = res1 + linear3->forward(bn3->forward(res1).relu());
    	print_variance("After second residual connection",res2);

    	torch::Tensor res3 = res2 + linear4->forward(bn4->forward(res2).relu());
    	print_variance("After third residual connection",res3);

    	torch::Tensor res4 = res3 + linear5->forward(bn5->forward(res3).relu());
    	print_variance("After fourth residual connection",res4);

    	torch::Tensor res5 = res4 + linear6->forward(bn6->forward(res4).relu());
    	print_variance("After fifth residual connection",res5);
    	return linear7->forward(res5);
    }
private:
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr}, linear4{nullptr},
	linear5{nullptr}, linear6{nullptr}, linear7{nullptr};
	torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr}, bn6{nullptr};
};

template<class T>
void run_one_step_of_model(T& model, torch::Tensor x_train, torch::Tensor y_train, torch::Device device) {

	auto loss_function = torch::nn::CrossEntropyLoss();

	// construct SGD optimizer and initialize learning rate and momentum
	auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.05).momentum(0.9));

    // load the data into a class that creates the batches
	int batch_size = 200;
	auto dataset = LRdataset(x_train.t(), y_train).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset), batch_size);

	// Initialize model weights
	model.apply(weights_init);
	model.to(device);

    // Get a batch
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
	    // Break out of this loop -- we just want to see the first
        // iteration, but usually we would continue
        break;
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
	std::cout << "// Define the ResidualNetwork model and run for one step\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	//# Define the model and run for one step
	//# Monitoring the variance at each point in the network
	int n_hidden = 100, n_input = 40, n_output = 10;
	ResidualNetwork model(n_input, n_output, n_hidden);
	run_one_step_of_model(model, X_train, y_train, device);

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Define the ResidualNetworkWithBatchNorm model and run for one step\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	ResidualNetworkWithBatchNorm bnmodel(n_input, n_output, n_hidden);
	run_one_step_of_model(bnmodel, X_train, y_train, device);

	std::cout << "Done!\n";
	return 0;
}



