/*
 * EncoderDecoderArchitecture.cpp
 *
 *  Created on: Jul 14, 2024
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

class EncoderImpl : public torch::nn::Module {
public:
	int n_features = 0;
	EncoderImpl() {}
	EncoderImpl(int _n_features, int _hidden_dim) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        hidden = torch::empty(0);
        basic_rnn = torch::nn::GRU(torch::nn::GRUOptions(n_features, hidden_dim).batch_first(true));
        register_module("basic_rnn", basic_rnn);
	}

	torch::Tensor forward(torch::Tensor X) {
    	torch::Tensor rnn_out;
        std::tie(rnn_out, hidden) = basic_rnn->forward(X);
        return rnn_out;
    }

private:
	int hidden_dim = 0;
    torch::Tensor hidden;
    torch::nn::GRU basic_rnn{nullptr};
};
TORCH_MODULE(Encoder);

class DecoderImpl : public torch::nn::Module {
public:
	torch::Tensor hidden;
	DecoderImpl() {}
	DecoderImpl(int _n_features, int _hidden_dim) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        hidden = torch::empty(0);
        //self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        basic_rnn = torch::nn::GRU(torch::nn::GRUOptions(n_features, hidden_dim).batch_first(true));
        register_module("basic_rnn", basic_rnn);
        regression = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, n_features));
        register_module("regression", regression);
	}

    void init_hidden(torch::Tensor hidden_seq) {
        // We only need the final hidden state
    	torch::Tensor hidden_final = hidden_seq.index({Slice(), Slice(-1, None)}); //[:, -1:] # N, 1, H
        // But we need to make it sequence-first
        hidden = hidden_final.permute({1, 0, 2}); //# 1, N, H
    }

    torch::Tensor forward(torch::Tensor X) {
        // X is N, 1, F
    	torch::Tensor batch_first_output;
        std::tie(batch_first_output, hidden) = basic_rnn->forward(X, hidden);

    	torch::Tensor last_output = batch_first_output.index({Slice(), Slice(-1, None)}); //[:, -1:]
		torch::Tensor out = regression->forward(last_output);

        // N, 1, F
        return out.view({-1, 1, n_features});
    }

private:
    int n_features = 0, hidden_dim = 0;
    torch::nn::GRU basic_rnn{nullptr};
    torch::nn::Linear regression{nullptr};
};
TORCH_MODULE(Decoder);

class EncoderDecoder : public torch::nn::Module {
public:
	torch::Tensor outputs;

	EncoderDecoder(Encoder _encoder, Decoder _decoder, int _input_len, int _target_len, float _teacher_forcing_prob=0.5) {
        encoder = _encoder;
        decoder = _decoder;
        input_len = _input_len;
        target_len = _target_len;
        teacher_forcing_prob = _teacher_forcing_prob;
        outputs = torch::empty(0);
        register_module("encoder", encoder);
        register_module("decoder", decoder);
	}

    void init_outputs(int batch_size) {
    	torch::Device device = encoder->parameters().begin()->device();
        // N, L (target), F
        outputs = torch::zeros({batch_size, target_len, encoder->n_features}).to(device);
    }

    void store_output(int i, torch::Tensor out) {
        // Stores the output
        outputs.index_put_({Slice(), Slice(i, i+1), Slice()}, out); //:, i:i+1, :], out)
    }

    torch::Tensor forward(torch::Tensor X) {
        // splits the data in source and target sequences
        // the target seq will be empty in testing mode
        // N, L, F
    	torch::Tensor source_seq = X.index({Slice(), Slice(None, input_len), Slice()}); //[:, :input_len, :]
    	torch::Tensor target_seq = X.index({Slice(), Slice(input_len, None), Slice()}); //X[:, self.input_len:, :]
        init_outputs(X.size(0));

        // Encoder expected N, L, F
        torch::Tensor hidden_seq = encoder->forward(source_seq);
        // Output is N, L, H
        decoder->init_hidden(hidden_seq);

        // The last input of the encoder is also
        // the first input of the decoder
        torch::Tensor dec_inputs = source_seq.index({Slice(), Slice(-1, None), Slice()}); //[:, -1:, :]

        // Generates as many outputs as the target length
        for(auto& i : range(target_len, 0)) {
            // Output of decoder is N, 1, F
        	torch::Tensor out = decoder->forward(dec_inputs);
            store_output(i, out);

            float prob = teacher_forcing_prob;
            // In evaluation/test the target sequence is
            // unknown, so we cannot use teacher forcing
            if( ! is_training() )
                prob = 0.;

            // If it is teacher forcing
            if( torch::rand(1).data().item<float>() <= prob) {
                // Takes the actual element
                dec_inputs = target_seq.index({Slice(), Slice(i, i+1), Slice()}); //[:, i:i+1, :]
            } else {
                // Otherwise uses the last predicted output
                dec_inputs = out;
            }
        }

        return outputs;
    }

private:
	Encoder encoder{nullptr};
	Decoder decoder{nullptr};
	int input_len = 0, target_len = 0;
	float teacher_forcing_prob = 0.;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Encoder\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor full_seq = torch::tensor({{-1, -1}, {-1, 1}, {1, 1}, {1, -1}}).to(torch::kFloat32).view({1, 4, 2});
	torch::Tensor source_seq = full_seq.index({Slice(), Slice(0, 2)}); //:, :2] // first two corners
	torch::Tensor target_seq = full_seq.index({Slice(), Slice(2, None)}); //[:, 2:] // last two corners

	torch::manual_seed(21);
	Encoder encoder = Encoder(2, 2);

	torch::Tensor hidden_seq = encoder->forward(source_seq); // # output is N, L, F
	torch::Tensor hidden_final = hidden_seq.index({Slice(), Slice(-1, None)}); //[:, -1:]   # takes last hidden state
	std::cout << "hidden_final:\n" << hidden_final << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Decoder\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(21);
	Decoder decoder = Decoder(2, 2);

	// Initial hidden state will be encoder's final hidden state
	decoder->init_hidden(hidden_seq);
	// Initial data point is the last element of source sequence
	torch::Tensor inputs = source_seq.index({Slice(), Slice(-1, None)}); //[:, -1:]

	int target_len = 2;
	for(auto& i : range(target_len, 0)) {
	    std::cout << "Hidden:\n" << decoder->hidden << '\n';
	    torch::Tensor out = decoder->forward(inputs);   // Predicts coordinates
	    //print(f'Output: {out}\n')
	    std::cout << "Output:\n" << out << '\n';
	    // Predicted coordinates are next step's inputs
	    inputs = out;
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Decoder - teacher forcing\n";
	std::cout << "// --------------------------------------------------\n";
	// Initial hidden state will be encoder's final hidden state
	decoder->init_hidden(hidden_seq);
	// Initial data point is the last element of source sequence
	inputs = source_seq.index({Slice(), Slice(-1, None)}); //[:, -1:]

	for(auto& i : range(target_len, 0)) {
		std::cout << "Teacher forcing hidden:\n" << decoder->hidden << '\n';
		torch::Tensor out = decoder->forward(inputs);   // Predicts coordinates
		std::cout << "Teacher forcing output:\n" << out << '\n';

		//But completely ignores the predictions and uses real data instead
		inputs = target_seq.index({Slice(), Slice(i, i+1)}); //[:, i:i+1]
	}

	// Initial hidden state will be encoder's final hidden state
	decoder->init_hidden(hidden_seq);
	// Initial data point is the last element of source sequence
	inputs = source_seq.index({Slice(), Slice(-1, None)}); //[:, -1:]

	float teacher_forcing_prob = 0.5;

	for(auto& i : range(target_len, 0)) {
		std::cout << "Teacher forcing prob hidden:\n" << decoder->hidden << '\n';
		torch::Tensor out = decoder->forward(inputs);   // Predicts coordinates
		//print(f'Output: {out}\n')
		std::cout << "Teacher forcing prob output:\n" << out << '\n';

		// If it is teacher forcing
		if( torch::rand(1).data().item<float>() <= teacher_forcing_prob) {
			// Takes the actual element
			//inputs = target_seq[:, i:i+1]
			inputs = target_seq.index({Slice(), Slice(i, i+1)});
		} else {
			// Otherwise uses the last predicted output
			inputs = out;
		}
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Encoder + Decoder\n";
	std::cout << "// --------------------------------------------------\n";
	EncoderDecoder encdec = EncoderDecoder(encoder, decoder, 2, 2, 1.0);
	encdec.train();
	std::cout << "encdec.forward(full_seq):\n" << encdec.forward(full_seq) << '\n';
	encdec.eval();
	std::cout << "encdec.forward(source_seq):\n" << encdec.forward(source_seq) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Data Preparation\n";
	std::cout << "// --------------------------------------------------\n";

	std::vector<torch::Tensor> points, test_points;
	torch::Tensor directions, test_directions;
	std::tie(points, directions) = generate_sequences(128, false, 13);

	torch::Tensor full_train = torch::stack(points, 0).to(torch::kFloat32);
	std::cout << "full_train:\n" << full_train.sizes() << '\n';

	torch::Tensor target_train = full_train.index({Slice(), Slice(2, None)}); //:, 2:]
	std::cout << "target_train:\n" << target_train.sizes() << '\n';

	std::tie(test_points, test_directions) = generate_sequences(128, false, 19);
	torch::Tensor full_test = torch::stack(test_points, 0).to(torch::kFloat32);
	torch::Tensor source_test = full_test.index({Slice(), Slice(None, 2)}); //[:, :2]
	torch::Tensor target_test = full_test.index({Slice(), Slice(2, None)}); //[:, 2:]

	auto train_data = LRdataset(full_train, target_train).map(torch::data::transforms::Stack<>());;
	auto test_data = LRdataset(source_test, target_test).map(torch::data::transforms::Stack<>());;

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
						         std::move(train_data), 16);
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
						         std::move(test_data), 16);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Model Training & Configuration\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(23);
	encoder = Encoder(2, 2);
	decoder = Decoder(2, 2);

	EncoderDecoder model = EncoderDecoder(encoder, decoder, 2, 2, 0.5);
	model.to(device);

	auto loss_fnc = torch::nn::MSELoss();
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
			auto loss = loss_fnc(output, y);
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
			auto loss = loss_fnc(output, y);
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
