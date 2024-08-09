/*
 * VariableLengthSequences.cpp
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

std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> pack_collate(std::vector<int> batch,
																	std::vector<torch::Tensor> points, torch::Tensor directs) {
	std::vector<torch::Tensor> X, ys; // = [item[0] for item in batch]
    //y = [item[1] for item in batch]
	for(auto& i : batch) {
		X.push_back(points[i]);
		ys.push_back(directs[i]);
	}
    //X_pack = rnn_utils.pack_sequence(X, enforce_sorted=False)
	torch::nn::utils::rnn::PackedSequence X_pack = torch::nn::utils::rnn::pack_sequence(X, false);
	torch::Tensor y = torch::cat(ys, 0).view({-1, 1});
	return std::make_tuple(X_pack, y);
}

class SquareModelPacked : public torch::nn::Module {
public:
	SquareModelPacked(int _n_features, int _hidden_dim, int _n_outputs) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        n_outputs = _n_outputs;
        hidden = torch::empty(0);
        cell = torch::empty(0);
        // Simple LSTM
        basic_rnn = torch::nn::LSTM(torch::nn::LSTMOptions(n_features, hidden_dim).bidirectional(true));
        register_module("basic_rnn", basic_rnn);
        // Classifier to produce as many logits as outputs
        classifier = torch::nn::Linear(torch::nn::LinearOptions(2 * hidden_dim, n_outputs));
        register_module("classifier", classifier);
	}

	torch::Tensor forward(torch::nn::utils::rnn::PackedSequence X) {
        // X is a PACKED sequence now
        // output is PACKED
        // final hidden state is (2, N, H) - bidirectional
        // final cell state is (2, N, H) - bidirectional
    	std::tuple<torch::nn::utils::rnn::PackedSequence,
				   std::tuple<torch::Tensor,torch::Tensor>> rt = basic_rnn->forward_with_packed_input(X);
    	torch::nn::utils::rnn::PackedSequence rnn_out = std::get<0>(rt);
    	std::tie(hidden, cell) = std::get<1>(rt);

        //rnn_out, (self.hidden, self.cell) = basic_rnn(X);
        // unpack the output (N, L, 2*H)
        torch::Tensor batch_first_output, seq_sizes;
        std::tie(batch_first_output, seq_sizes) = torch::nn::utils::rnn::pad_packed_sequence(rnn_out, true); //batch_first=True

        // only last item in sequence (N, 1, 2*H)
        torch::Tensor seq_idx = torch::arange({seq_sizes.size(0)});
        torch::Tensor last_output = batch_first_output.index({seq_idx, (seq_sizes-1)}); //batch_first_output[seq_idx, seq_sizes-1];
        // classifier will output (N, 1, n_outputs)
		torch::Tensor out = classifier->forward(last_output);

        // final output is (N, n_outputs)
        return out.view({-1, n_outputs});
	}
private:
	int n_features = 0, hidden_dim = 0, n_outputs = 0;
	torch::nn::LSTM basic_rnn{nullptr};
	torch::nn::Linear classifier{nullptr};
	torch::Tensor hidden, cell;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::vector<torch::Tensor> points;
	torch::Tensor directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Variable-Length Sequences\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor s0 = points[0];      // 4 data points
	torch::Tensor s1 = points[1].index({Slice(2, None)}); //[1][2:];  // 2 data points
	torch::Tensor s2 = points[2].index({Slice(1, None)});  // 3 data points
	std::cout << "s0: " << s0.sizes() << " s1: " << s1.sizes() << " s2: " << s2.sizes() << '\n';

	// Padding
	std::vector<torch::Tensor> seq_tensors = {s0, s1, s2};
	torch::Tensor padded = torch::nn::utils::rnn::pad_sequence(seq_tensors, true); //batch_first = true
	std::cout << "padded:\n" << padded << "\n";

	torch::manual_seed(11);
	torch::nn::RNN rnn = torch::nn::RNN(torch::nn::RNNOptions(2, 2).batch_first(true));
	torch::Tensor output_padded, hidden_padded;
	std::tie(output_padded, hidden_padded) = rnn->forward(padded);
	std::cout << "output_padded:\n" << output_padded << "\n";

	std::cout << "hidden_padded.permute(1, 0, 2):\n" << hidden_padded.permute({1, 0, 2}) << "\n";

	// Packing
	torch::nn::utils::rnn::PackedSequence packed = torch::nn::utils::rnn::pack_sequence(seq_tensors, false); // enforce_sorted=false
	std::cout << "packed data:\n" << packed.data() << "\nbatch_sizes:\n"
			  << packed.batch_sizes() << "\nsorted_indices:\n"
			  << packed.sorted_indices() << "\nunsorted_indices:\n"
			  << packed.unsorted_indices() << '\n';

	torch::Tensor index = torch::tensor({0, 3, 6, 8});
	std::cout << "(packed.data[[0, 3, 6, 8]] == seq_tensors[0]).all(): " <<
			(packed.data().index_select(0, index) == seq_tensors[0]).all() << '\n';

	std::cout << padded.sizes() << " -- " << packed.data().unsqueeze(0).sizes() << '\n';
	std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> rt = rnn->forward_with_packed_input(packed);
	torch::nn::utils::rnn::PackedSequence output_packed = std::get<0>(rt);
	torch::Tensor hidden_packed = std::get<1>(rt);

	std::cout << "output_packed:\n" << output_packed.data() << "\n";
	std::cout << "hidden_packed:\n" << hidden_packed << "\n";
	std::cout << "hidden_padded:\n" << hidden_padded << "\n";

	std::cout << "hidden_packed == hidden_padded:\n" <<
				(hidden_packed == hidden_padded) << '\n';

	index = torch::tensor({2, 5});
	// x1 sequence
	std::cout << "output_packed.data[[2, 5]]:\n" << output_packed.data().index_select(0, index) << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Unpacking (to padded)\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor output_unpacked, seq_sizes;
	std::tie(output_unpacked, seq_sizes) = torch::nn::utils::rnn::pad_packed_sequence(output_packed, true); //batch_first
	std::cout << "output_unpacked:\n" << output_unpacked << "\n";
	std::cout << "seq_sizes:\n" << seq_sizes << "\n";

	std::cout << "output_unpacked[:, -1]:\n" << output_unpacked.index({Slice(), -1}) << "\n";
	torch::Tensor seq_idx = torch::arange(seq_sizes.size(0));
	std::cout << "output_unpacked[seq_idx, seq_sizes-1]:\n" << output_unpacked.index({seq_idx, seq_sizes-1}) << "\n";

	// Packing (from padded)
	std::vector<int> seqs_len;
	for(auto& seq : seq_tensors)
		seqs_len.push_back(seq.size(0));

	torch::Tensor len_seqs = torch::tensor(seqs_len).to(torch::kLong);
	std::cout << "len_seqs:\n" << len_seqs << "\n";
	std::cout << "padded:\n" << padded.sizes() << "\n";
	/*
	 * input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]),
	 * B is the batch size, and * is any number of dimensions (including 0). If batch_first is true,
	 * B x T x * input is expected.
	 *
	 * inline PackedSequence torch::nn::utils::rnn::pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first = false, bool enforce_sorted = true)
	 */
	packed = torch::nn::utils::rnn::pack_padded_sequence(padded, len_seqs, true, false);
	std::cout << "packed data:\n" << packed.data() << "\nbatch_sizes:\n"
			  << packed.batch_sizes() << "\nsorted_indices:\n"
			  << packed.sorted_indices() << "\nunsorted_indices:\n"
			  << packed.unsorted_indices() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Variable-Length Dataset\n";
	std::cout << "// --------------------------------------------------\n";

	std::vector<torch::Tensor> var_points;
	torch::Tensor var_directions;
	std::tie(var_points, var_directions) = generate_sequences(128, true, 19);
	std::cout << "var_points[:2]:\n" << var_points[0]<< "\n" << var_points[1] << '\n';

	for(int i = 0; i < var_points.size(); i++)
		var_points[i] = var_points[i].to(torch::kFloat32);

	var_directions = var_directions.to(torch::kFloat32).view({-1,1});
	std::cout << "var_directions: " << var_directions.sizes() << '\n';

	std::vector<torch::Tensor> test_points;
	torch::Tensor test_directions;
	std::tie(test_points, test_directions) = generate_sequences(128, false, 19);
	for(int i = 0; i < test_points.size(); i++)
		test_points[i] = test_points[i].to(torch::kFloat32);

	test_directions = test_directions.to(torch::kFloat32).view({-1,1});

	int batch_size = 16;

	std::list<std::vector<int>> batchs = data_index_iter(static_cast<int>(var_directions.size(0)), batch_size, true);

	for( auto& batch : batchs ) {
		std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> dt = pack_collate(batch, var_points, var_directions);
		torch::nn::utils::rnn::PackedSequence X = std::get<0>(dt);
		torch::Tensor y = std::get<1>(dt);
		std::cout << "X data:\n" << X.data() << "\nbatch_sizes:\n"
				  << X.batch_sizes() << "\nsorted_indices:\n"
				  << X.sorted_indices() << "\nunsorted_indices:\n"
				  << X.unsorted_indices() << '\n';

		std::cout << "y:" << y.sizes() << "\n";

		//auto output = model.forward(X);
		//std::cout << "output:" << output.sizes() << "\n";
		break;
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Model Configuration & Training\n";
	std::cout << "// --------------------------------------------------\n";

	torch::manual_seed(21);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	SquareModelPacked model = SquareModelPacked(2, 2, 1);
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

		std::list<std::vector<int>> batchs = data_index_iter(static_cast<int>(var_directions.size(0)), 16, true);

		for (auto &batch : batchs) {
			std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> dt =
					pack_collate(batch, var_points, var_directions);
			torch::nn::utils::rnn::PackedSequence X = std::get<0>(dt).to(device);
			torch::Tensor y = std::get<1>(dt).to(device);

			auto output = model.forward(X);
			auto loss = loss_fn(output, y);
			optimizer.zero_grad();           		// clear gradients for this training step
			loss.backward();						// backpropagation, compute gradients
			optimizer.step();						// apply gradients
			epoch_train_loss += loss.data().item<float>();
		    num_train_samples += X.data().size(0);
		}

		model.eval();

		std::list<std::vector<int>> test_batchs = data_index_iter(static_cast<int>(test_directions.size(0)), batch_size, false);

		for (auto &batch : test_batchs) {

			std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> dt =
					pack_collate(batch, test_points, test_directions);
			torch::nn::utils::rnn::PackedSequence X = std::get<0>(dt).to(device);
			torch::Tensor y = std::get<1>(dt).to(device);

			auto output = model.forward(X);
			auto loss = loss_fn(output, y);
			epoch_test_loss += loss.data().item<float>();
			num_test_samples += X.data().size(0);
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




