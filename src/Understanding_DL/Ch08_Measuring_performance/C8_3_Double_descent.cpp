/*
 * C8_3_Double_descent.cpp
 *
 *  Created on: Dec 21, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <cmath>

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

// Return an initialized model with two hidden layers and n_hidden hidden units at each
torch::nn::Sequential get_model(int n_hidden) {

	int D_i = 40;    		// Input dimensions
    int D_k = n_hidden;		// Hidden dimensions
    int D_o = 10;    		// Output dimensions

    // Define a model with two hidden layers
    // And ReLU activations between them
	torch::nn::Sequential model = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(D_i, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_k)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(D_k, D_o)));

    // Call the function you just defined
	weights_init(model);

  // Return the model
  return model;
}

std::tuple<double, double> fit_model(torch::nn::Sequential model,
		torch::Tensor x_train, torch::Tensor y_train, torch::Tensor x_test,
		torch::Tensor y_test, int n_epoch, torch::Device device, int64_t h_variable) {

	// choose cross entropy loss function (equation 5.24)
	auto loss_function = torch::nn::CrossEntropyLoss();

	// construct SGD optimizer and initialize learning rate and momentum
	auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

	// store the loss and the % correct at each epoch
	std::vector<double> errors_train, errors_test;

	int batch_size = 100;
	auto dataset = LRdataset(x_train, y_train).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset), batch_size);

	auto dataset_ts = LRdataset(x_test, y_test).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset_ts), batch_size);

	double tr_error = 0., ts_error =0.;
	int tr_num_smaples = 0, ts_num_smaples = 0;
	int tr_correct = 0, ts_correct = 0;

	model->train();
	for(auto& epoch : range(n_epoch, 1)) {
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

		    //tr_ls += loss.data().item<float>();
		    torch::Tensor _, predicted_train_class;
			std::tie(_, predicted_train_class) = torch::max(pred.data(), 1);
		    tr_correct += torch::sum((predicted_train_class == y_batch).to(torch::kInt)).data().item<int>();
		    tr_num_smaples += x_batch.size(0);
		}


		model->eval();

		for (auto &batch : *test_loader) {
			// retrieve inputs and labels for this batch
			auto x_batch = batch.data.to(device);
			auto y_batch = batch.target.flatten().to(device);
			auto pred = model->forward(x_batch);

			// compute the loss
			auto loss = loss_function(pred, y_batch);
			torch::Tensor _, predicted_train_class;
			std::tie(_, predicted_train_class) = torch::max(pred.data(), 1);
			ts_correct += torch::sum((predicted_train_class == y_batch).to(torch::kInt)).data().item<int>();
			ts_num_smaples += x_batch.size(0);
		}
	}
	tr_error = 100. - (tr_correct*1.0/tr_num_smaples)*100;
	ts_error = 100. - (ts_correct*1.0/tr_num_smaples)*100;
	printf("Hidden variables: %3ld, train error %3.2f, test error: %3.2f\n", h_variable, tr_error, ts_error);

    return std::make_tuple(tr_error, ts_error);
}

int64_t count_parameters(torch::nn::Sequential model) {
	int64_t pcnt = 0;
    for(auto& p : model->parameters()) {
    	if( p.requires_grad() ) {
    		pcnt += p.numel();
    	}
    }
    return pcnt;
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

	int cnt = 0;
	// Add 15% noise to training labels
	for(auto& c_y : range(static_cast<int>(y_train.size(0)), 0) ) {
	    float random_number = torch::rand({1}).data().item<float>();
	    if(random_number < 0.15 ) {
	        int64_t random_int = torch::randint(0, 10, {1}).data().item<long>();
	        y_train[c_y] = random_int;
	        cnt += 1;
	    }
	}
	std::cout << "cnt: " << cnt << " y_train.size(0): " << y_train.size(0) << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Run whole dataset to get statistics\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	std::vector<int> hidden_variables = {2,8,18,30,45,60,90,140,200,400};

	std::vector<double>  errors_train_all, errors_test_all;
	torch::Tensor total_weights_all = torch::zeros({static_cast<int>(hidden_variables.size())}).to(torch::kInt64);

	// loop over the dataset n_epoch times
	int n_epoch = 1000;

	// For each hidden variable size
	int idx = 0;
	for(auto& c_hidden : hidden_variables) {
	    std::cout << "Training model with " << c_hidden << " hidden variable size.\n";
	    // Get a model
	    torch::nn::Sequential model = get_model(c_hidden) ;
	    // Count and store number of weights
	    total_weights_all[idx] = count_parameters(model);

	    // Train the model
	    double errors_train, errors_test;
	    std::tie(errors_train, errors_test) = fit_model(model, X_train,
	    		y_train, X_test, y_test, n_epoch, device, c_hidden);

	    // Store the results
	    errors_train_all.push_back(errors_train);
	    errors_test_all.push_back(errors_test);
	    idx++;
	}

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Plot the results\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Assuming data['y'] is available and contains the training examples
	int num_training_examples = y_train.size(0);

	// Find the index where total_weights_all is closest to num_training_examples
	auto closest_index = torch::argmin(torch::abs(total_weights_all - num_training_examples));

	// Get the corresponding value of hidden variables
	int hidden_variable_at_num_training_examples = hidden_variables[closest_index.data().item<int>()];

	// Plot the results
	auto F = figure(true);
	F->size(800, 600);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);
	matplot::plot(fx, hidden_variables, errors_train_all, "m-")->line_width(2).display_name("train");
	matplot::plot(fx, hidden_variables, errors_test_all, "b--")->line_width(2).display_name("test");

	// Add a vertical line at the point where total weights equal the number of training examples
	matplot::plot(fx, std::vector<double> {hidden_variable_at_num_training_examples*1.0, hidden_variable_at_num_training_examples*1.0},
			std::vector<double> {0, 100}, "g-:")->line_width(3); //.display_name("N(weights) = N(train)");
	auto [ta, la] = matplot::textarrow(fx, hidden_variable_at_num_training_examples*1.0 + 20, 10.0,
			hidden_variable_at_num_training_examples*1.0, 10.0, "N(weights) = N(train)");
	auto [tb, lb] = matplot::textarrow(fx, hidden_variables[8]*1.0 + 20, errors_train_all[8] + 10,
			hidden_variables[8]*1.0, errors_train_all[8], "train");
	auto [tc, lc] = matplot::textarrow(fx, hidden_variables[8]*1.0 + 20, errors_test_all[8] + 10,
			hidden_variables[8]*1.0, errors_test_all[8], "test");
	ta->color("red").font_size(14);
	la->color("k");
	tb->color("red").font_size(14);
	lb->color("k");
	tc->color("red").font_size(14);
	lc->color("k");
	matplot::ylim(fx, {0, 100});
	matplot::xlabel(fx, "No. hidden variables");
	matplot::ylabel(fx, "Error");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





