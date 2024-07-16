/*
 * RNN_and_LSTM.cpp
 *
 *  Created on: Jul 2, 2024
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

class SquareModel : public torch::nn::Module {
public:
	SquareModel(int _n_features, int _hidden_dim, int _n_outputs) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        n_outputs = _n_outputs;
        hidden = torch::empty(0);
        // Simple RNN
        basic_rnn = torch::nn::RNN(torch::nn::RNNOptions(n_features, hidden_dim).batch_first(true));

        // Classifier to produce as many logits as outputs
        classifier = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, n_outputs));
        register_module("basic_rnn", basic_rnn);
        register_module("classifier", classifier);
	}

	torch::Tensor forward(torch::Tensor X) {
        // X is batch first (N, L, F)
        // output is (N, L, H)
        // final hidden state is (1, N, H)
		torch::Tensor batch_first_output;
        std::tie(batch_first_output, hidden) = basic_rnn->forward(X);

        // only last item in sequence (N, 1, H)
        torch::Tensor last_output = batch_first_output.index({Slice(), -1});

		// classifier will output (N, 1, n_outputs)
		torch::Tensor out = classifier->forward(last_output);

        // final output is (N, n_outputs)
        return out.view({-1, n_outputs});
    }

private:
	int hidden_dim = 1, n_features = 2, n_outputs = 2;
	torch::nn::RNN basic_rnn{nullptr};
	torch::nn::Linear classifier{nullptr};
	torch::Tensor hidden;
};

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// RNN Cell\n";
	std::cout << "// --------------------------------------------------\n";

	int n_features = 2, hidden_dim = 2;

	torch::nn::RNNCell rnn_cell = torch::nn::RNNCell(torch::nn::RNNCellOptions(n_features, hidden_dim));
	auto rnn_state = rnn_cell->named_parameters(); //state_dict()
	std::cout << rnn_state.size() << '\n';

	torch::nn::Linear linear_input = torch::nn::Linear(torch::nn::LinearOptions(n_features, hidden_dim));
	torch::nn::Linear linear_hidden = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim));

	{
		torch::NoGradGuard no_grad;
		linear_input->weight = rnn_state["weight_ih"].clone();
	    linear_input->bias = rnn_state["bias_ih"].clone();
	    linear_hidden->weight = rnn_state["weight_hh"].clone();
	    linear_hidden->bias = rnn_state["bias_hh"].clone();
	}

	torch::Tensor initial_hidden = torch::zeros({1, hidden_dim});
	std::cout << "initial_hidden:\n" << initial_hidden << '\n';
	torch::Tensor th = linear_hidden->forward(initial_hidden);
	std::cout << "th:\n" << th << '\n';

	torch::Tensor bases = torch::randint(4, 10);
	std::cout << "bases:\n" << bases << '\n';

	torch::Tensor basic_corners = torch::tensor({{-1, -1}, {-1, 1}, {1, 1}, {1, -1}});
	std::cout << "basic_corners[-1]:\n" << basic_corners.flip({0}) << '\n';

	std::vector<torch::Tensor> points;
	torch::Tensor directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);
	torch::Tensor X = points[0];
	std::cout << "X:\n" << X << '\n';
	torch::Tensor tx = linear_input->forward(X.index({Slice(0,1), Slice()}));
	std::cout << "tx:\n" << tx << '\n';

	torch::Tensor adding = th + tx;
	std::cout << "adding:\n" << adding << '\n';
	std::cout << "torch::tanh(adding):\n" << torch::tanh(adding) << '\n';
	std::cout << "rnn_cell(X[0:1]):\n" << rnn_cell->forward(X.index({Slice(0,1), Slice()})) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//WRONG feed the full sequence to the RNN cell\n";
	std::cout << "// --------------------------------------------------\n";
	std::cout << "rnn_cell(X:\n" << rnn_cell->forward(X) << '\n';

	/*
	To effectively use the RNN cell in a sequence, we need to loop over the data points
	and provide the updated hidden state at each step:
	*/
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Effectively use the RNN cell in a sequence\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor hidden = torch::zeros({1, hidden_dim});
	for(auto& i : range(static_cast<int>(X.size(0)), 0)) {
	    auto out = rnn_cell->forward(X.index({Slice(i, i+1), Slice()}), hidden);
	    std::cout << "out:\n" << out << '\n';
	    hidden = out;
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//RNN Layer\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(19);
	torch::nn::RNN rnn = torch::nn::RNN(torch::nn::RNNOptions(n_features, hidden_dim));

	rnn_state = rnn->named_parameters();
	for(auto& k : rnn_state.keys() ) {
		std::cout << k << " " << rnn_state[k].sizes() << '\n';
	}

	// Shapes
	std::vector<torch::Tensor> ppts = {points[0], points[1], points[2]};
	torch::Tensor batch = torch::stack(ppts, 0);
	std::cout << "batch.shape: " << batch.sizes() << '\n';
	torch::Tensor permuted_batch = batch.permute({1, 0, 2});
	std::cout << "permuted_batch.shape: " << permuted_batch.sizes() << '\n';

	rnn = torch::nn::RNN(torch::nn::RNNOptions(n_features, hidden_dim));
	torch::Tensor out, final_hidden;
	std::tie(out, final_hidden) = rnn->forward(permuted_batch);
	std::cout << "rnn => out.shape: " << out.sizes() << '\n';
	std::cout << "rnn => final_hidden.shape: " << final_hidden.sizes() << '\n';
	std::cout << "out:\n" << out << '\n';
	std::cout << "out[-1]:\n" << out[-1] << '\n';

	std::cout << "(out[-1] == final_hidden).all(): " << (out[-1] == final_hidden).all() << '\n';
	torch::Tensor batch_hidden = final_hidden.permute({1, 0, 2});
	std::cout << "batch.shape: " << batch.sizes() << '\n';

	torch::nn::RNN rnn_batch_first = torch::nn::RNN(torch::nn::RNNOptions(n_features, hidden_dim).batch_first(true));
	std::tie(out, final_hidden) = rnn_batch_first->forward(batch);
	std::cout << "rnn_batch_first => out.shape: " << out.sizes() << '\n';
	std::cout << "rnn_batch_first => final_hidden.shape: " << final_hidden.sizes() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Stacked RNN\n";
	std::cout << "// --------------------------------------------------\n";
	torch::nn::RNN rnn_stacked = torch::nn::RNN(torch::nn::RNNOptions(2, 2).num_layers(2).batch_first(true));
	auto state = rnn_stacked->named_parameters();
	for(auto& k : state.keys() ) {
		std::cout << k << "\n" << state[k] << '\n';
	}

	auto weights = rnn_stacked->all_weights();
	int cnt = 0;
	for(auto& w : weights ) {
		std::cout << "cnt: " << cnt << "\n" << w << '\n';
		cnt++;
	}

	torch::nn::RNN rnn_layer0 = torch::nn::RNN(torch::nn::RNNOptions(2, 2).batch_first(true));
	torch::nn::RNN rnn_layer1 = torch::nn::RNN(torch::nn::RNNOptions(2, 2).batch_first(true));

	{
		torch::NoGradGuard no_grad;
		rnn_layer0->all_weights().clear();
		for(auto& i :range(4, 0))
			rnn_layer0->all_weights().push_back(weights[i].clone());

		rnn_layer1->all_weights().clear();
		for(auto& i :range(4, 4))
			rnn_layer1->all_weights().push_back(weights[i].clone());

	}

	torch::Tensor x = points[0].unsqueeze(0), out0, h0, out1, h1;
	std::cout << "x:\n" << x.sizes() << '\n';
	std::tie(out0, h0) = rnn_layer0->forward(x);
	std::tie(out1, h1) = rnn_layer1->forward(out0);

	std::cout << "out1:\n" << out << '\n';
	std::cout << "torch.cat([h0, h1]):\n" << torch::cat({h0, h1}) << '\n';
	std::tie(out, hidden) = rnn_stacked->forward(x);
	std::cout << "out:\n" << out << '\n';
	std::cout << "hidden:\n" << hidden << '\n';

	std::cout << "(out.index({Slice(), -1}) == hidden.permute({1, 0, 2}).index({Slice(), -1})).all(): "
			  << (out.index({Slice(), -1}) == hidden.permute({1, 0, 2}).index({Slice(), -1})).all() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Bidirectional RNN\n";
	std::cout << "// --------------------------------------------------\n";
	torch::nn::RNN rnn_bidirect = torch::nn::RNN(torch::nn::RNNOptions(2, 2).bidirectional(true).batch_first(true));

	state = rnn_bidirect->named_parameters();
	for(auto& k : state.keys() ) {
		std::cout << k << "\n" << state[k] << '\n';
	}
	torch::nn::RNN rnn_forward = torch::nn::RNN(torch::nn::RNNOptions(2, 2).batch_first(true));
	torch::nn::RNN rnn_reverse = torch::nn::RNN(torch::nn::RNNOptions(2, 2).batch_first(true));

	weights = rnn_bidirect->all_weights();
	//rnn_forward.load_state_dict(dict(list(state.items())[:4]))
	//rnn_reverse.load_state_dict(dict([(k[:-8], v) for k, v in list(state.items())[4:]]))
	{
		torch::NoGradGuard no_grad;
		rnn_forward->all_weights().clear();
		for(auto& i :range(4, 0))
			rnn_forward->all_weights().push_back(weights[i].clone());

		rnn_reverse->all_weights().clear();
		for(auto& i :range(4, 4))
			rnn_reverse->all_weights().push_back(weights[i].clone());
	}

	torch::Tensor x_rev = torch::flip(x, {1}); // N, L, F
	std::cout << "x_rev:\n" << x_rev << '\n';
	torch::Tensor h, out_rev, h_rev;
	std::tie(out, h) = rnn_forward->forward(x);
	std::tie(out_rev, h_rev) = rnn_reverse->forward(x_rev);

	torch::Tensor out_rev_back = torch::flip(out_rev, {1});
	std::cout << "torch::cat({out, out_rev_back}, 2):\n" << torch::cat({out, out_rev_back}, 2) << '\n';
	std::cout << "torch::cat({h, h_rev}):\n" << torch::cat({h, h_rev}) << '\n';

	std::tie(out, hidden) = rnn_bidirect->forward(x);
	std::cout << "out:\n" << out << '\n';
	std::cout << "hidden:\n" << hidden << '\n';

	torch::Tensor is_eq = (out.index({Slice(), -1}) == hidden.permute({1, 0, 2}).view({1, -1}));
	std::cout << "is_eq:\n" << is_eq << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Square Model\n";
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

	SquareModel model = SquareModel(2, 2, 1);
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

		 printf("Epoch: %3d | train loss: %.4f, | valid loss: %.4f\n", (epoch+1),
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


