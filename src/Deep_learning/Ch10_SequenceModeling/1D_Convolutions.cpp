/*
 * 1D_Convolutions.cpp
 *
 *  Created on: Jul 4, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

namespace F = torch::nn::functional;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::vector<torch::Tensor> points;
	torch::Tensor directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//1D Convolution\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor temperatures = torch::tensor({5., 11., 15., 6., 5., 3., 3., 0., 0., 3., 4., 2., 1.});

	int size = 5;
	torch::Tensor weight = torch::ones(size) * 0.2;
	std::cout << "F::conv1d(temperatures.view({1, 1, -1}),  weight.view({1, 1, -1})):\n"
			  << F::conv1d(temperatures.view({1, 1, -1}),  weight.view({1, 1, -1})) << "\n";

	// Shapes
	torch::Tensor seqs = torch::stack(points, 0).to(torch::kFloat32); // N, L, F
	torch::Tensor seqs_length_last = seqs.permute({0, 2, 1});
	std::cout << "seqs_length_last.shape: " << seqs_length_last.sizes() << "\n"; // N, F=C, L

	// Multiple Features or Channels
	torch::manual_seed(17);
	torch::nn::Conv1d conv_seq = torch::nn::Conv1d(torch::nn::Conv1dOptions(2, 1, 2).bias(false));

	std::cout << "conv_seq.weight.shape: " << conv_seq->weight.sizes() << "\n" << conv_seq->weight << '\n';
	std::cout << "conv_seq(seqs_length_last[0:1]):\n" << conv_seq->forward(seqs_length_last[0]) << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Dilation\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(17);
	torch::nn::Conv1d conv_dilated = torch::nn::Conv1d(torch::nn::Conv1dOptions(2, 1, 2).dilation(2).bias(false));

	std::cout << "conv_dilated.weight.shape: " << conv_dilated->weight.sizes() << "\n" << conv_dilated->weight << '\n';
	std::cout << "conv_dilated(seqs_length_last[0:1]):\n" << conv_dilated->forward(seqs_length_last[0]) << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Model Configuration & Training\n";
	std::cout << "// --------------------------------------------------\n";

	std::vector<torch::Tensor> test_points;
	torch::Tensor test_directions;
	std::tie(test_points, test_directions) = generate_sequences(128, false, 19);
	int batch_size = 16;

	torch::Tensor train_data = torch::stack(points, 0).permute({0, 2, 1}).to(torch::kFloat32);
	torch::Tensor train_d = directions.view({-1, 1}).to(torch::kFloat32);

	torch::Tensor test_data = torch::stack(test_points, 0).permute({0, 2, 1}).to(torch::kFloat32);
	torch::Tensor test_d = test_directions.view({-1, 1}).to(torch::kFloat32);
	std::cout << train_data.sizes() << " " << train_d.sizes() << '\n';

	auto train_dataset = LRdataset(train_data, train_d).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(train_dataset), batch_size);

	auto tst_dataset = LRdataset(test_data, test_d)
					   .map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(tst_dataset), batch_size);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(21);
	torch::nn::Sequential model = torch::nn::Sequential();
	model->push_back("conv1d", torch::nn::Conv1d(torch::nn::Conv1dOptions(2, 1, 2)));
	model->push_back("relu", torch::nn::ReLU());
	model->push_back("flatten", torch::nn::Flatten());
	model->push_back("output", torch::nn::Linear(torch::nn::LinearOptions(3, 1)));
	model->to(device);
	auto loss_fn = torch::nn::BCEWithLogitsLoss();
	auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

	int epochs = 100;
	std::vector<double> train_loss, test_loss, xx;

	for(int  epoch = 0; epoch < epochs; epoch++ ) {
		model->train();

		double epoch_train_loss = 0.0;
		double epoch_test_loss = 0.0;
		int64_t num_train_samples = 0;
		int64_t num_test_samples = 0;

		for (auto &batch : *train_loader) {

			auto X = batch.data.to(device);
			auto y = batch.target.to(device);

			auto output = model->forward(X);
			auto loss = loss_fn(output, y);
			optimizer.zero_grad();           		// clear gradients for this training step
			loss.backward();						// backpropagation, compute gradients
			optimizer.step();						// apply gradients
			epoch_train_loss += loss.data().item<float>();
		    num_train_samples += X.size(0);
		}

		model->eval();

		for (auto &batch : *test_loader) {

			auto X = batch.data.to(device);
			auto y = batch.target.to(device);
			auto output = model->forward(X);
			auto loss = loss_fn(output, y);
			epoch_test_loss += loss.data().item<float>();
			num_test_samples += X.size(0);
		}

		train_loss.push_back(epoch_train_loss*1.0/num_train_samples);
		test_loss.push_back(epoch_test_loss*1.0/num_test_samples);
		xx.push_back((epoch + 1)*1.0);

		 printf("Epoch: %3d | train loss: %.5f, | valid loss: %.5f\n", (epoch+1),
				 (epoch_train_loss*1.0/num_train_samples),(epoch_test_loss*1.0/num_test_samples));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xx, train_loss, "b")->line_width(2);
	matplot::plot(ax1, xx, test_loss, "r:")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "Epochs");
	matplot::ylabel(ax1, "Loss");
	matplot::legend(ax1, {"Train loss", "Validation loss"});
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}







