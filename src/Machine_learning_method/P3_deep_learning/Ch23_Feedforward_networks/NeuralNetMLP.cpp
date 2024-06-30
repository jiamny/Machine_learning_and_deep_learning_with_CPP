/*
 * NeuralNetMLP.cpp
 *
 *  Created on: Jun 21, 2024
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

#include "../../../Utils/fashion.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/activation.h"
#include "../../../Utils/loss.h"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor int_to_onehot(torch::Tensor y, int num_labels) {

	torch::Tensor ary = torch::zeros({y.size(0), num_labels}, torch::kInt32).to(y.device());
    for(auto& i : range(static_cast<int>(y.size(0)), 0)) {
    	int val = y[i].data().item<int>();
        ary[i][val] = 1;
    }

    return ary;
}

struct NeuralNetMLP {

	NeuralNetMLP(int _num_features, int _num_hidden, int _num_classes, torch::Device device=torch::kCPU, int random_seed=123) {

        num_classes = _num_classes;
        num_features = _num_features;
        num_hidden =_num_hidden;

        // hidden
        torch::manual_seed(random_seed);

        weight_h = torch::normal(0.0, 0.1, {num_hidden, num_features}).to(device);
            //loc=0.0, scale=0.1, size=(num_hidden, num_features))
        bias_h = torch::zeros({num_hidden}, torch::kFloat32).to(device);

        // output
        weight_out = torch::normal(0.0, 0.1, {num_classes, num_hidden}).to(device);
            //loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        bias_out = torch::zeros({num_classes}, torch::kFloat32).to(device);
	}


	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Hidden layer
        // input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        // output dim: [n_examples, n_hidden]
		torch::Tensor z_h = torch::mm(x, weight_h.t()) + bias_h;
		torch::Tensor a_h = torch::sigmoid(z_h);

        // Output layer
        // input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        // output dim: [n_examples, n_classes]
		torch::Tensor z_out = torch::mm(a_h, weight_out.t()) + bias_out;
		torch::Tensor a_out = torch::sigmoid(z_out);
        return std::make_pair(a_h, a_out);
	}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward(
    		torch::Tensor x, torch::Tensor a_h, torch::Tensor a_out, torch::Tensor y) {

        //#########################
        //### Output layer weights
        //#########################

        //# onehot encoding
    	torch::Tensor y_onehot = int_to_onehot(y, num_classes);

        //# Part 1: dLoss/dOutWeights
        //## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        //## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        //## for convenient re-use

        //# input/output dim: [n_examples, n_classes]
    	torch::Tensor d_loss__d_a_out = 2.*(a_out - y_onehot) / y.size(0);

        // input/output dim: [n_examples, n_classes]
    	torch::Tensor d_a_out__d_z_out = a_out * (1. - a_out); // # sigmoid derivative

        //# output dim: [n_examples, n_classes]
    	torch::Tensor delta_out = d_loss__d_a_out * d_a_out__d_z_out; // # "delta (rule) placeholder"

        //# gradient for output weights

        //# [n_examples, n_hidden]
    	torch::Tensor d_z_out__dw_out = a_h;

        //# input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        //# output dim: [n_classes, n_hidden]
    	torch::Tensor d_loss__dw_out = torch::mm(delta_out.t(), d_z_out__dw_out);
    	c10::OptionalArrayRef<long int> dim = {0};
		torch::Tensor d_loss__db_out = torch::sum(delta_out, dim);


        //#################################
        //# Part 2: dLoss/dHiddenWeights
        //## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight

        //# [n_classes, n_hidden]
		torch::Tensor d_z_out__a_h = weight_out;

        //# output dim: [n_examples, n_hidden]
		torch::Tensor d_loss__a_h = torch::mm(delta_out, d_z_out__a_h);

        //# [n_examples, n_hidden]
		torch::Tensor d_a_h__d_z_h = a_h * (1. - a_h); // # sigmoid derivative

        //# [n_examples, n_features]
		torch::Tensor d_z_h__d_w_h = x;

        //# output dim: [n_hidden, n_features]
		torch::Tensor d_loss__d_w_h = torch::mm((d_loss__a_h * d_a_h__d_z_h).t(), d_z_h__d_w_h);
		c10::OptionalArrayRef<long int> dm = {0};
		torch::Tensor d_loss__d_b_h = torch::sum((d_loss__a_h * d_a_h__d_z_h), dm);

        return std::make_tuple(d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h);
    }

	int num_classes = 0, num_hidden = 0, num_features = 0;
	torch::Tensor weight_h, weight_out, bias_h, bias_out;
};


template<typename T>
std::pair<double, double> compute_mse_and_acc(NeuralNetMLP nnet, T& loader, int num_labels=10, torch::Device device=torch::kCPU) {
    double mse = 0.;
    int correct_pred = 0, num_examples = 0;

    int cnt = 0;
    for (auto& batch : *loader) {
    	auto features = batch.data.squeeze().to(device);
    	features = features.reshape({features.size(0), -1});
    	auto targets = batch.target.to(device);
    	torch::Tensor _, probas;
        std::tie(_, probas) = nnet.forward(features);
        std::optional<long int> dim = {1};
        torch::Tensor predicted_labels = torch::argmax(probas, dim);

		torch::Tensor onehot_targets = int_to_onehot(targets, num_labels);
		torch::Tensor loss = torch::mean(torch::pow((onehot_targets - probas), 2));
		torch::Tensor t = torch::sum((predicted_labels == targets).cpu().to(torch::kInt32));
        correct_pred += t.data().item<int>();

        num_examples += targets.size(0);
        mse += loss.cpu().data().item<double>();
        cnt++;
    }

    mse = mse/cnt;
    double acc = correct_pred*1.0/num_examples;
    return std::make_pair( mse, acc );
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';


    // Create an unordered_map to hold label names
    std::map<int, std::string> fashionMap = {
    		{0, "T-shirt/top"},
			{1, "Trouser"},
			{2, "Pullover"},
			{3, "Dress"},
			{4, "Coat"},
			{5, "Sandal"},
			{6, "Short"},
			{7, "Sneaker"},
			{8, "Bag"},
			{9, "Ankle boot"}};

	int num_epochs = 50; // train the training data n times, to save time, we just train 1 epoch
	int batch_size = 100;
	float learning_rate = 0.1;

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	FASHION train_datas(FASHION_data_path, FASHION::Mode::kTrain);
	auto train_dataset = train_datas.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);

	FASHION valid_datas(FASHION_data_path, FASHION::Mode::kTest);
	auto valid_dataset = valid_datas.map(torch::data::transforms::Stack<>());
	auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
					         std::move(valid_dataset), batch_size);

	NeuralNetMLP model(28*28, 50, 10, device);
/*
    torch::Tensor X = torch::randn({100, 784}).to(device);
    torch::Tensor y = torch::randint(0, 9, {100}).to(device);
    torch::Tensor a_h, a_out;

    std::tie(a_h, a_out) = model.forward(X);

    std::cout << a_h << '\n';
    std::cout << a_out << '\n';

	torch::Tensor d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h;
	std::tie(d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h) =
			model.backward(X, a_h, a_out, y);
*/

    std::vector<double> epoch_loss;
    std::vector<double> epoch_train_acc;
	std::vector<double> epoch_valid_acc;
	std::vector<double> epoch_s;

	for(auto& epoch : range(num_epochs, 0)) {

		torch::AutoGradMode enable_grad(true);

		for (auto& batch : *train_loader) {
			auto b_x = batch.data.squeeze().to(device);
			b_x = b_x.reshape({b_x.size(0), -1});
			auto b_y = batch.target.to(device);
			//std::cout << b_x.sizes() << " " << b_y.sizes() << '\n';
			torch::Tensor a_h, a_out;
			std::tie(a_h, a_out) = model.forward(b_x);

			// #### Compute gradients ####
			torch::Tensor d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h;
			std::tie(d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h) =
					model.backward(b_x, a_h, a_out, b_y);

			// #### Update weights ####
			model.weight_h -= learning_rate * d_loss__d_w_h;
			model.bias_h -= learning_rate * d_loss__d_b_h;
			model.weight_out -= learning_rate * d_loss__d_w_out;
			model.bias_out -= learning_rate * d_loss__d_b_out;
		}

		// #### Epoch Logging ####
		torch::NoGradGuard no_grad;

		double train_mse, train_acc, valid_mse, valid_acc;
		std::tie(train_mse, train_acc) = compute_mse_and_acc(model, train_loader, 10, device);
		std::tie(valid_mse, valid_acc) = compute_mse_and_acc(model, valid_loader, 10, device);

        train_acc = train_acc*100;
        valid_acc = valid_acc*100;
        printf("Epoch: %03d/%03d | Train MSE: %.3f | Train Acc: %.2f | Valid Acc: %.2f\n",
        		(epoch+1), num_epochs, train_mse, train_acc, valid_acc);

        epoch_train_acc.push_back(train_acc);
        epoch_valid_acc.push_back(valid_acc);
        epoch_loss.push_back(train_mse);
        epoch_s.push_back((epoch+1)*1.0);
	}

	auto F = figure(true);
	F->size(1400, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 2);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	//matplot::ylim(ax1, {0.2, 0.99});
	matplot::plot(ax1, epoch_s, epoch_loss, "m-:")->line_width(2).display_name("Train loss");
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "MSE loss");
	matplot::legend(ax1, {});

	auto ax2 = F->nexttile();
	matplot::hold(ax2, true);
	matplot::plot(ax2, epoch_s, epoch_train_acc, "b-")->line_width(2).display_name("Train acc");
	matplot::plot(ax2, epoch_s,  epoch_valid_acc, "r-.")->line_width(2).display_name("Valid acc");
    matplot::hold(ax2, false);
    matplot::xlabel(ax2, "epoch");
    matplot::legend(ax2, {});
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}

