/*
 * LongShortTermMemory.cpp
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

torch::Tensor forget_gate(torch::nn::Linear f_hidden, torch::nn::Linear f_input, torch::Tensor h, torch::Tensor x) {
	torch::Tensor thf = f_hidden->forward(h);
	torch::Tensor txf = f_input(x);
	torch::Tensor f = torch::sigmoid(thf + txf);
    return f;  // red
}

torch::Tensor output_gate(torch::nn::Linear o_hidden, torch::nn::Linear o_input, torch::Tensor h, torch::Tensor x) {
	torch::Tensor tho = o_hidden->forward(h);
	torch::Tensor txo = o_input->forward(x);
	torch::Tensor o = torch::sigmoid(tho + txo);
    return o;  // blue
}

torch::Tensor input_gate(torch::nn::Linear i_hidden, torch::nn::Linear i_input, torch::Tensor h, torch::Tensor x) {
	torch::Tensor thi = i_hidden->forward(h);
	torch::Tensor txi = i_input->forward(x);
	torch::Tensor i = torch::sigmoid(thi + txi);
    return i;  // green
}

class SquareModelLSTM : public torch::nn::Module {
public:
	SquareModelLSTM(int _n_features, int _hidden_dim, int _n_outputs) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        n_outputs = _n_outputs;
        hidden = torch::empty(0);
        cell = torch::empty(0);
        // Simple LSTM
        basic_lstm = torch::nn::LSTM(torch::nn::LSTMOptions(n_features, hidden_dim).batch_first(true));
        // Classifier to produce as many logits as outputs
        classifier = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, n_outputs));
        register_module("basic_lstm", basic_lstm);
        register_module("classifier", classifier);
	}

    torch::Tensor forward(torch::Tensor X) {
        // X is batch first (N, L, F)
        // output is (N, L, H)
        // final hidden state is (1, N, H)
        // final cell state is (1, N, H)
    	torch::Tensor batch_first_output;
    	std::tuple<torch::Tensor, torch::Tensor> t = std::make_tuple(hidden, cell);
    	std::tie(batch_first_output, t) = basic_lstm->forward(X);

        // only last item in sequence (N, 1, H)
    	torch::Tensor last_output = batch_first_output.index({Slice(), -1}); //[:, -1];
        // classifier will output (N, 1, n_outputs)
    	torch::Tensor out = classifier->forward(last_output);

        // final output is (N, n_outputs)
        return out.view({-1, n_outputs});
    }

private:
	int n_features = 0, hidden_dim = 0, n_outputs = 0;
	torch::nn::LSTM basic_lstm{nullptr};
	torch::nn::Linear classifier{nullptr};
	torch::Tensor hidden, cell;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// LSTM Cell\n";
	std::cout << "// --------------------------------------------------\n";

	torch::manual_seed(17);

	int n_features = 2, hidden_dim = 2;
	torch::nn::LSTMCell lstm_cell = torch::nn::LSTMCell(torch::nn::LSTMCellOptions(n_features, hidden_dim));
	auto lstm_state = lstm_cell->named_parameters();

	torch::Tensor Wx = lstm_state["weight_ih"], bx = lstm_state["bias_ih"];
	torch::Tensor Wh = lstm_state["weight_hh"], bh = lstm_state["bias_hh"];

	std::cout << "Wx: " << Wx.sizes() << " Wh: " << Wh.sizes() << '\n';
	std::cout << "bx: " << bx.sizes() << " bh: " << bh.sizes() << '\n';

	// Split weights and biases for data points
	std::vector<torch::Tensor> sWx = torch::split(Wx, hidden_dim, 0);
	torch::Tensor Wxi = sWx[0], Wxf = sWx[1], Wxg = sWx[2], Wxo = sWx[3];
	std::vector<torch::Tensor> sbx = torch::split(bx, hidden_dim, 0);
	torch::Tensor bxi = sbx[0], bxf = sbx[1], bxg = sbx[2], bxo = sbx[3];

	// Split weights and biases for hidden state
	std::vector<torch::Tensor> sWh = torch::split(Wh, hidden_dim, 0);
	torch::Tensor Whi = sWh[0], Whf = sWh[1], Whg = sWh[2], Who = sWh[3];
	std::vector<torch::Tensor> sbh = torch::split(bh, hidden_dim, 0);
	torch::Tensor bhi = sbh[0], bhf = sbh[1], bhg = sbh[2],  bho = sbh[3];

	std::cout << "Wxi:\n" << Wxi << "\nbxi:\n" << bxi << '\n';

	// Creates linear layers for the components
	torch::nn::Linear i_hidden{nullptr}, i_input{nullptr}, f_hidden{nullptr},
					  f_input{nullptr}, o_hidden{nullptr}, o_input{nullptr};

	std::tie(i_hidden, i_input) = linear_layers(Wxi, bxi, Whi, bhi); // input gate - green
	std::tie(f_hidden, f_input) = linear_layers(Wxf, bxf, Whf, bhf); // forget gate - red
	std::tie(o_hidden, o_input) = linear_layers(Wxo, bxo, Who, bho); // output gate - blue

	torch::nn::RNNCell g_cell = torch::nn::RNNCell(torch::nn::RNNCellOptions(n_features, hidden_dim)); // black
	{
		torch::NoGradGuard no_grad;
		g_cell->bias_ih = bxg.clone();
		g_cell->weight_ih = Wxg.clone();
		g_cell->bias_hh = bhg.clone();
		g_cell->weight_hh = Whg.clone();
	}

	torch::Tensor initial_hidden = torch::zeros({1, hidden_dim});
	torch::Tensor initial_cell = torch::zeros({1, hidden_dim});

	std::vector<torch::Tensor> points;
	torch::Tensor directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);

	torch::Tensor X = points[0];
	torch::Tensor first_corner = X.index({Slice(0,1), Slice()});

	torch::Tensor g = g_cell->forward(first_corner);
	torch::Tensor i = input_gate(i_hidden, i_input, initial_hidden, first_corner);
	torch::Tensor gated_input = g * i;
	std::cout << "gated_input:\n" << gated_input << '\n';

	torch::Tensor f = forget_gate(f_hidden, f_input, initial_hidden, first_corner);
	torch::Tensor gated_cell = initial_cell * f;
	std::cout << "gated_cell:\n" << gated_cell << '\n';

	torch::Tensor c_prime = gated_cell + gated_input;
	std::cout << "c_prime:\n" << c_prime << '\n';

	torch::Tensor o = output_gate(o_hidden, o_input, initial_hidden, first_corner);
	torch::Tensor h_prime = o * torch::tanh(c_prime);
	std::cout << "h_prime:\n" << h_prime << '\n';

	std::tie(h_prime, c_prime) = lstm_cell->forward(first_corner);
	std::cout << "h_prime:\n" << h_prime << '\n';
	std::cout << "c_prime:\n" << c_prime << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Model Configuration & Training\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(21);

	std::vector<torch::Tensor> test_points;
	torch::Tensor test_directions;
	std::tie(test_points, test_directions) = generate_sequences(128, false, 19);
	int batch_size = 16;

	torch::Tensor train_data = torch::stack(points, 0).to(torch::kFloat32);
	torch::Tensor train_d = directions.view({-1, 1}).to(torch::kFloat32);

	torch::Tensor test_data = torch::stack(test_points, 0).to(torch::kFloat32);
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

	SquareModelLSTM model = SquareModelLSTM(2, 2, 1);
	model.to(device);

	auto loss_fn = torch::nn::BCEWithLogitsLoss();
	auto optimizer = torch::optim::Adam(model.parameters(), 0.01);

	int epochs = 100;
	std::vector<double> train_loss, test_loss, xx;

	for(int  epoch = 0; epoch < epochs; epoch++ ) {
		model.train();

		double epoch_train_loss = 0.0;
		double epoch_test_loss = 0.0;
		int64_t num_train_samples = 0;
		int64_t num_test_samples = 0;

		for (auto &batch : *train_loader) {

			auto X = batch.data.to(device);
			auto y = batch.target.to(device);

			auto output = model.forward(X);
			auto loss = loss_fn(output, y);
			optimizer.zero_grad();           		// clear gradients for this training step
			loss.backward();						// backpropagation, compute gradients
			optimizer.step();						// apply gradients
			epoch_train_loss += loss.data().item<float>();
		    num_train_samples += X.size(0);
		}

		model.eval();

		for (auto &batch : *test_loader) {

			auto X = batch.data.to(device);
			auto y = batch.target.to(device);
			auto output = model.forward(X);
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




