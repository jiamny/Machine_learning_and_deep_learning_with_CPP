/*
 * GatedRecurrentUnits.cpp
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


torch::Tensor reset_gate(torch::nn::Linear r_hidden, torch::nn::Linear r_input,
						 torch::Tensor h, torch::Tensor x) {
	torch::Tensor thr = r_hidden->forward(h);
	torch::Tensor txr = r_input->forward(x);
	torch::Tensor r = torch::sigmoid(thr + txr);
    return r;  // red
}

torch::Tensor update_gate(torch::nn::Linear z_hidden, torch::nn::Linear z_input,
						  torch::Tensor h, torch::Tensor x) {
	torch::Tensor thz = z_hidden->forward(h);
	torch::Tensor txz = z_input->forward(x);
	torch::Tensor z = torch::sigmoid(thz + txz);
    return z;   // blue
}

torch::Tensor candidate_n(torch::nn::Linear n_hidden, torch::nn::Linear n_input,
						  torch::Tensor h, torch::Tensor x, torch::Tensor r) {
	torch::Tensor thn = n_hidden(h);
	torch::Tensor txn = n_input(x);
	torch::Tensor n = torch::tanh(r * thn + txn);
    return n;  // black
}

class SquareModelGRU : public torch::nn::Module {
public:
	SquareModelGRU(int _n_features, int _hidden_dim, int _n_outputs) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        n_outputs = _n_outputs;
        hidden = torch::empty(0);
        // Simple GRU
        basic_gru = torch::nn::GRU(torch::nn::GRUOptions(n_features, hidden_dim).batch_first(true));
        // Classifier to produce as many logits as outputs
        classifier = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, n_outputs));
        register_module("basic_gru", basic_gru);
        register_module("classifier", classifier);
	}

	torch::Tensor forward(torch::Tensor X) {
        // X is batch first (N, L, F)
        // output is (N, L, H)
        // final hidden state is (1, N, H)
		torch::Tensor batch_first_output;
        std::tie(batch_first_output, hidden) = basic_gru->forward(X);

        // only last item in sequence (N, 1, H)
        torch::Tensor last_output = batch_first_output.index({Slice(), -1});

		// classifier will output (N, 1, n_outputs)
		torch::Tensor out = classifier->forward(last_output);

        // final output is (N, n_outputs)
        return out.view({-1, n_outputs});
	}

private:
	int n_features = 0, hidden_dim = 0, n_outputs = 0;
	torch::nn::GRU basic_gru{nullptr};
	torch::nn::Linear classifier{nullptr};
	torch::Tensor hidden;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// GRU Cell\n";
	std::cout << "// --------------------------------------------------\n";

	torch::manual_seed(17);

	int n_features = 2, hidden_dim = 2;

	torch::nn::GRUCell gru_cell = torch::nn::GRUCell(torch::nn::GRUCellOptions(n_features, hidden_dim));
	auto gru_state = gru_cell->named_parameters();

	torch::Tensor Wx = gru_state["weight_ih"], bx = gru_state["bias_ih"];
	torch::Tensor Wh = gru_state["weight_hh"], bh = gru_state["bias_hh"];

	std::cout << "Wx: " << Wx.sizes() << " Wh: " << Wh.sizes() << '\n';
	std::cout << "bx: " << bx.sizes() << " bh: " << bh.sizes() << '\n';

	std::vector<torch::Tensor> sWx = torch::split(Wx, hidden_dim, 0);
	torch::Tensor Wxr = sWx[0], Wxz =sWx[1], Wxn = sWx[2];
	std::vector<torch::Tensor> sbx = torch::split(bx, hidden_dim, 0);
	torch::Tensor bxr = sbx[0], bxz = sbx[1], bxn = sbx[2];

	std::vector<torch::Tensor> sWh = torch::split(Wh, hidden_dim, 0);
	torch::Tensor Whr = sWh[0], Whz = sWh[1], Whn = sWh[2];
	std::vector<torch::Tensor> sbh = torch::split(bh, hidden_dim, 0);
	torch::Tensor bhr = sbh[0], bhz = sbh[1], bhn = sbh[2];

	std::cout << "Wxr:\n" << Wxr << "\nbxr:\n" << bxr << '\n';

	torch::nn::Linear r_hidden{nullptr}, r_input{nullptr}, z_hidden{nullptr},
					  z_input{nullptr}, n_hidden{nullptr}, n_input{nullptr};

	std::tie(r_hidden, r_input) = linear_layers(Wxr, bxr, Whr, bhr);	 // reset gate - red
	std::tie(z_hidden, z_input) = linear_layers(Wxz, bxz, Whz, bhz);	 // update gate - blue
	std::tie(n_hidden, n_input) = linear_layers(Wxn, bxn, Whn, bhn); 	 // candidate state - black

	std::vector<torch::Tensor> points;
	torch::Tensor directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);

	torch::Tensor initial_hidden = torch::zeros({1, hidden_dim});
	torch::Tensor X = points[0];
	torch::Tensor first_corner = X.index({Slice(0,1), Slice()});

	torch::Tensor r = reset_gate(r_hidden, r_input, initial_hidden, first_corner);
	std::cout << "r:\n" << r << '\n';

	torch::Tensor n = candidate_n(n_hidden, n_input, initial_hidden, first_corner, r);
	std::cout << "n:\n" << n << '\n';

	torch::Tensor z = update_gate(z_hidden, z_input, initial_hidden, first_corner);
	std::cout << "z:\n" << z << '\n';

	torch::Tensor h_prime = n*(1-z) + initial_hidden*z;
	std::cout << "h_prime:\n" << h_prime << '\n';

	std::cout << "gru_cell(first_corner):\n" << gru_cell->forward(first_corner) << '\n';

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

	SquareModelGRU model = SquareModelGRU(2, 2, 1);
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



