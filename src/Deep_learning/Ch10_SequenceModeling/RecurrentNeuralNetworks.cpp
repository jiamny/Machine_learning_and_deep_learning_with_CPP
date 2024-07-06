/*
 * RecurrentNeuralNetworks.cpp
 *
 *  Created on: Jul 1, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/fashion.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

class RNN : public torch::nn::Module {
public:
	/*
	   hidden_size - rnn hidden unit
       num_layers  - number of rnn layer
       batch_first - input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
	 */
	RNN(int INPUT_SIZE, int hidden_size=64, int num_layers=1, bool batch_first=true) {
		// if use nn.RNN(), it hardly learns
        rnn = torch::nn::LSTM(torch::nn::LSTMOptions(INPUT_SIZE, hidden_size).num_layers(num_layers).batch_first(batch_first));
        out = torch::nn::Linear(torch::nn::LinearOptions(64, 10));
        register_module("rnn", rnn);
        register_module("out", out);
	}

	torch::Tensor forward(torch::Tensor x) {
        // x shape (batch, time_step, input_size)
        // r_out shape (batch, time_step, output_size)
        // h_n shape (n_layers, batch, hidden_size)
        // h_c shape (n_layers, batch, hidden_size)
		torch::Tensor h_n, h_c, r_out;
		std::tuple<torch::Tensor, torch::Tensor> state = {h_n, h_c};

        std::tie(r_out, state) = rnn->forward(x, None);   // None represents zero initial hidden state

        // choose r_out at the last time step
        torch::Tensor ot = out->forward(r_out.index({Slice(), -1, Slice()})); //:, -1, :]);
        return ot;
	}

private:
	torch::nn::Linear out{nullptr};
	torch::nn::LSTM rnn{nullptr};
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Hidden recurrence versus output recurrence\n";
	std::cout << "// --------------------------------------------------\n";
	torch::nn::RNN rnn_layer = torch::nn::RNN(torch::nn::RNNOptions(5, 2).num_layers(1).batch_first(true));
	torch::OrderedDict<std::string, torch::Tensor> params = rnn_layer->named_parameters(false);
	torch::Tensor w_xh = params["weight_ih_l0"];
	torch::Tensor w_hh = params["weight_hh_l0"];
	torch::Tensor b_xh = params["bias_ih_l0"];
	torch::Tensor b_hh = params["bias_hh_l0"];

	std::cout << "W_xh shape: " << w_xh.sizes() << '\n' << w_xh << '\n';
	std::cout << "W_hh shape: " << w_hh.sizes() << '\n' << w_hh << '\n';
	std::cout << "b_xh shape: " << b_xh.sizes() << '\n' << b_xh << '\n';
	std::cout << "b_hh shape: " << b_hh.sizes() << '\n' << b_hh << '\n';

	torch::Tensor x_seq = torch::tensor({{1.0, 1., 1.,1., 1.,},
										{2.0, 2.0, 2., 2., 2.},
										{3.0, 3., 3., 3., 3.}}, torch::kFloat32);

	// output of the simple RNN:
	torch::Tensor output, hn, prev_h;
	std::tie(output, hn) = rnn_layer(torch::reshape(x_seq, {1, 3, 5}));
	std::cout << "after rnn\n";

	//## manually computing the output:
	std::vector<torch::Tensor> out_man;
	for(auto& t : range(3, 0)) {
		torch::Tensor xt = torch::reshape(x_seq[t], {1, 5});
	    std::cout << "Time step => " << t << "\n";
	    std::cout << "   Input           : ";
	    printVector(tensorTovector(xt.to(torch::kDouble)));

	    torch::Tensor ht = torch::matmul(xt, torch::transpose(w_xh, 0, 1)) + b_xh;
	    std::cout << "   Hidden          : ";
	    printVector(tensorTovector(ht.detach().to(torch::kDouble)));

	    if( t > 0 ) {
	        prev_h = out_man[t-1];
	    } else {
	        prev_h = torch::zeros(ht.sizes(), torch::kFloat32);
	    }

	    torch::Tensor ot = ht + torch::matmul(prev_h, torch::transpose(w_hh, 0, 1)) + b_hh;
	    ot = torch::tanh(ot);
	    out_man.push_back(ot);
	    std::cout << "   Output (manual) : ";
	    printVector(tensorTovector(ot.detach().to(torch::kDouble)));

		std::cout << "   RNN output      : ";
		printVector(tensorTovector(output.index({Slice(), t}).detach().to(torch::kDouble)));

	    std::cout << "\n";
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// RNN classifier\n";
	std::cout << "// --------------------------------------------------\n";

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

    bool show_image = true;

	// Hyper Parameters
	int EPOCH = 20;               // train the training data n times, to save time, we just train 1 epoch
	int BATCH_SIZE = 64;
	int TIME_STEP = 28;          // rnn time step / image height
	int INPUT_SIZE = 28;         // rnn input size / image width
	float LR = 0.01;             // learning rate

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
					         std::move(train_dataset), BATCH_SIZE);

	FASHION test_data(FASHION_data_path, FASHION::Mode::kTest);
	at::Tensor test_x = train_datas.images().index({Slice(0, 2000), Slice()});
	torch::Tensor test_y = train_datas.targets().index({Slice(0, 2000)});

	if (show_image ) {
		at::Tensor im = train_datas.images()[0].squeeze();
		torch::Tensor target = train_datas.targets()[0].squeeze();

		int type_id = target.data().item<int>();
		std::cout << "type_id = " << type_id << " name = " << fashionMap.at(type_id) << "\n";

		std::vector<std::vector<double>>  C = get_mnist_image(im);
		auto F = figure(true);
		F->size(600, 600);
		F->add_axes(false);
		F->reactive_mode(false);
		F->tiledlayout(1, 1);
		F->position(0, 0);

		auto ax = F->nexttile();
		matplot::image(ax, C);
		auto search = fashionMap.find(type_id);
		if (search != fashionMap.end()) {
			std::string it = fashionMap[type_id];
			matplot::title(ax, it.c_str());
		}
		matplot::show();
	}

	RNN rnn(INPUT_SIZE);
	std::cout << rnn << '\n';
	rnn.to(device);

	torch::Tensor Xi = torch::randn({64, 1, 28, 28}).to(device);
	torch::Tensor ot = rnn.forward(Xi.squeeze());

	auto optimizer = torch::optim::Adam(rnn.parameters(), LR);   // optimize all cnn parameters
	auto loss_func = torch::nn::CrossEntropyLoss();               // the target label is not one-hotted

	for(auto& epoch : range(EPOCH, 0)) {

		rnn.train();
		float ls = 0.0;
		int step = 0;
		for (auto& batch : *train_loader) {

		    // Transfer images and target labels to device
			auto b_x = batch.data.to(device);
		    auto b_y = batch.target.to(device);

	        auto output = rnn.forward(b_x.squeeze());			// rnn output
	        auto loss = loss_func(output, b_y);		// cross entropy loss
	        optimizer.zero_grad();           		// clear gradients for this training step
	        loss.backward();						// backpropagation, compute gradients
	        optimizer.step();						// apply gradients
	        ls += loss.data().item<float>();
	        step += 1;
		}

		rnn.eval();
    	torch::Tensor test_output = rnn.forward(test_x.squeeze().to(device));
        std::optional<long int>dim = {1};
		torch::Tensor pred_y = torch::argmax(test_output, dim);
        torch::Tensor accuracy = torch::sum(pred_y == test_y.to(device)) *1.0 / test_y.size(0);
        printf("Epoch: %2d | train loss: %.4f, | test accuracy: %.3f\n", (epoch+1),
        		(ls/step), accuracy.data().item<float>());
	}

	// print 10 predictions from test data
	torch::Tensor test_output = rnn.forward(test_x.to(device).index({Slice(0, 20), Slice()}).view({-1, 28, 28}));
	torch::Tensor pred_y = std::get<1>(torch::max(test_output, 1)).cpu().data().squeeze();
	std::cout << pred_y.sizes() << '\n';

	std::cout << "prediction item:\n";
	printVector(tensorTovector(pred_y.to(torch::kDouble)));

	std::cout << "real item:\n";
	printVector(tensorTovector(test_y.index({Slice(0, 20)}).to(torch::kDouble)));

	std::cout << "Done!\n";
	return 0;
}

