/*
 * Attention.cpp
 *
 *  Created on: Jul 18, 2024
 *      Author: jiamny
 */
#include <iostream>
#include <unistd.h>

#include "Encoder_decoder.h"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor calc_attention_scores(torch::Tensor ks, torch::Tensor q) {
    int dims = q.size(-1);
    // N, 1, H x N, H, L -> N, 1, L
    torch::Tensor products = torch::bmm(q, ks.permute({0, 2, 1}));
    torch::Tensor scaled_products = products / std::sqrt(dims);
    // -------------------------------------------------------------------------------------
    // To transform alignment scores into attention scores we can use the softmax function:
    // -------------------------------------------------------------------------------------
    torch::Tensor alphas = F::softmax(scaled_products, -1);
    return alphas;
}

class AttentionImpl : public torch::nn::Module {
public:
	AttentionImpl(int hidden_dim, int _input_dim=0, bool _proj_values=false) {

        d_k = hidden_dim;

        if( _input_dim == 0) {
        	input_dim = hidden_dim;
        } else {
        	input_dim = _input_dim;
        }

        proj_values = _proj_values;

		// Affine transformations for Q, K, and V
        linear_query = torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim));
        linear_key = torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim));
        linear_value = torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim));
        alphas = torch::empty(0);
        register_module("linear_query", linear_query);
        register_module("linear_key", linear_key);
        register_module("linear_value", linear_value);
	}

    void init_keys(torch::Tensor _keys) {
        keys = _keys;
        proj_keys = linear_key->forward(keys);
        if( proj_values ) {
      	  values = linear_value->forward(keys);
        } else {
      	  values = keys;
        }
    }

    torch::Tensor score_function(torch::Tensor query) {
    	torch::Tensor proj_query = linear_query->forward(query);
        // scaled dot product
        // N, 1, H x N, H, L -> N, 1, L
    	torch::Tensor dot_products = torch::bmm(proj_query, proj_keys.permute({0, 2, 1}));
    	torch::Tensor scores =  dot_products / std::sqrt(d_k);
        return scores;
    }

    torch::Tensor forward(torch::Tensor query, torch::Tensor mask=torch::empty(0)) {
        // Query is batch-first N, 1, H
    	torch::Tensor scores = score_function(query); // N, 1, L
        if( mask.numel() != 0 ) {
            scores = scores.masked_fill(mask == 0, -1e9);
        }
        torch::Tensor _alphas = F::softmax(scores, -1); // N, 1, L
        alphas = _alphas.detach();

        // N, 1, L x N, L, H -> N, 1, H
        torch::Tensor context = torch::bmm(alphas, values);
        return context;
    }

	int d_k = 0, input_dim = 0;
	torch::Tensor keys, alphas, proj_keys, values;
	bool proj_values =false;
	torch::nn::Linear linear_query{nullptr}, linear_key{nullptr}, linear_value{nullptr};
};
TORCH_MODULE(Attention);

class DecoderAttnImpl : public torch::nn::Module {
public:
	int hidden_dim = 0, n_features = 0;
	torch::Tensor hidden;
	torch::nn::GRU basic_rnn{nullptr};
	Attention attn{nullptr};
	torch::nn::Linear regression{nullptr};

	DecoderAttnImpl(int _n_features, int _hidden_dim) {
        hidden_dim = _hidden_dim;
        n_features = _n_features;
        hidden = torch::empty(0);
        basic_rnn = torch::nn::GRU(torch::nn::GRUOptions(n_features, hidden_dim).batch_first(true));
        attn = Attention(hidden_dim);
        regression = torch::nn::Linear(torch::nn::LinearOptions(2 * hidden_dim, n_features));

        register_module("basic_rnn", basic_rnn);
        register_module("attn", attn);
        register_module("regression", regression);
	}

    void init_hidden(torch::Tensor hidden_seq) {
        // the output of the encoder is N, L, H
        // and init_keys expects batch-first as well
        attn->init_keys(hidden_seq);
        torch::Tensor hidden_final = hidden_seq.index({Slice(), Slice(-1, None)}); //, -1:]
        hidden = hidden_final.permute({1, 0, 2});   // L, N, H
    }

    torch::Tensor forward(torch::Tensor X, torch::Tensor mask=torch::empty(0)) {
        // X is N, 1, F
    	torch::Tensor batch_first_output;
        std::tie(batch_first_output, hidden) = basic_rnn->forward(X, hidden);

        torch::Tensor query = batch_first_output.index({Slice(), Slice(-1, None)}); //[:, -1:]
        // Attention
		torch::Tensor context = attn->forward(query, mask);
		torch::Tensor concatenated = torch::cat({context, query}, -1);
		torch::Tensor out = regression->forward(concatenated);

        // N, 1, F
        return out.view({-1, 1, n_features});
    }
};
TORCH_MODULE(DecoderAttn);

class EncoderDecoderAttnImpl : public torch::nn::Module {
public:
	torch::Tensor outputs, alphas;
	Encoder encoder{nullptr};
	DecoderAttn decoder{nullptr};
	int input_len = 0, target_len = 0;
	float teacher_forcing_prob = 0.;

	EncoderDecoderAttnImpl(Encoder _encoder, DecoderAttn _decoder, int _input_len, int _target_len, float _teacher_forcing_prob=0.5) {
        encoder = _encoder;
        decoder = _decoder;
        input_len = _input_len;
        target_len = _target_len;
        teacher_forcing_prob = _teacher_forcing_prob;
        outputs = torch::empty(0);
        alphas = torch::empty(0);
        register_module("encoder", encoder);
        register_module("decoder", decoder);
	}

    void init_outputs(int batch_size) {
    	torch::Device device = encoder->parameters().begin()->device();
        // N, L (target), F
        outputs = torch::zeros({batch_size, target_len, encoder->n_features}).to(device);

        // N, L (target), L (source)
        alphas = torch::zeros({batch_size, target_len, input_len}).to(device);
    }

    void store_output(int i, torch::Tensor out) {
        // Stores the output
        outputs.index_put_({Slice(), Slice(i, i+1), Slice()}, out); //:, i:i+1, :], out)
        alphas.index_put_({Slice(), Slice(i, i+1), Slice()}, decoder->attn->alphas);
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
};
TORCH_MODULE(EncoderDecoderAttn);


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
	/*
	 * The encoder’s hidden states are used both as "keys" (K)"values" (V).
	 * the decoder’s hidden state is called a "query" (Q)
	 *
	 * The resulting multiplication of a "value" by its corresponding attention score is called an alignment vector.
	 * And, as you can see in the diagram, the sum of all alignment vectors (that is, the weighted average of the
	 * hidden states) is called a context vector
	 *
	 * The attention scores are based on matching each hidden state of the decoder (h2) to every
	 * hidden state of the encoder (h0 and h1).
	 *
	 * The "query" (Q) is matched to both "keys" (K) to compute the attention scores (s) used to compute
	 * the context vector, which is simply the weighted average of the "values" (V).
	 */

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Computing the Context Vector\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor full_seq = torch::tensor({{-1, -1}, {-1, 1}, {1, 1}, {1, -1}}, torch::kFloat32).view({1, 4, 2});
	torch::Tensor source_seq = full_seq.index({Slice(), Slice(0, 2)}); //:, :2]
	torch::Tensor target_seq = full_seq.index({Slice(), Slice(2, None)});
	torch::manual_seed(21);
	Encoder encoder(2, 2);
	torch::Tensor hidden_seq = encoder->forward(source_seq);
	torch::Tensor values = hidden_seq; // N, L, H
	std::cout << "Values:\n" << values << '\n';

	torch::Tensor keys = hidden_seq; // N, L, H
	std::cout << "Keys:\n" << keys << '\n';

	Decoder decoder(2, 2);
	decoder->init_hidden(hidden_seq);

	torch::Tensor inputs = source_seq.index({Slice(), Slice(-1, None)}); //[:, -1:];
	torch::Tensor out = decoder->forward(inputs);
	torch::Tensor query = decoder->hidden.permute({1, 0, 2});  // N, 1, H
	std::cout << "Query:\n" << query << '\n';

	torch::Tensor alphas = calc_attention_scores(keys, query);
	std::cout << "Attention_scores:\n" << alphas << '\n';

	// N, 1, L x N, L, H -> 1, L x L, H -> 1, H
	torch::Tensor context_vector = torch::bmm(alphas, values);
	std::cout << "Context_vector:\n" << context_vector << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Attention Mechanism\n";
	std::cout << "// --------------------------------------------------\n";

	// Source Mask
	source_seq = torch::tensor({{{-1., 1.}, {0., 0.}}});
	// pretend there's an encoder here...
	keys = torch::tensor({{{-.38, .44}, {.85, -.05}}});
 	query = torch::tensor({{{-1., 1.}}});
 	torch::Tensor source_mask = (source_seq != 0).all(2).unsqueeze(1);
 	std::cout << "source_mask:\n" << source_mask << '\n'; // N, 1, L

 	torch::manual_seed(11);
 	Attention attnh = Attention(2);
 	attnh->init_keys(keys);

 	torch::Tensor context = attnh->forward(query, source_mask);
	std::cout << "attnh->alphas:\n" << attnh->alphas << '\n';
	std::cout << "context:\n" << context << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//DecoderAttn\n";
	std::cout << "// --------------------------------------------------\n";

	source_seq = full_seq.index({Slice(), Slice(0, 2)}); //:, :2]
	target_seq = full_seq.index({Slice(), Slice(2, None)}); //[:, 2:]

	torch::manual_seed(21);
	encoder = Encoder(2, 2);
	DecoderAttn decoder_attn = DecoderAttn(2, 2);

	// Generates hidden states (keys and values)
	hidden_seq = encoder->forward(source_seq);
	decoder_attn->init_hidden(hidden_seq);

	// Target sequence generation
	inputs = source_seq.index({Slice(), Slice(-1, None)}); //[:, -1:]
	int target_len = 2;
	for(auto& i : range(target_len, 0)) {
	    out = decoder_attn->forward(inputs);
	    std::cout << "Output: " << out << '\n';
	    inputs = out;
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//Encoder + Decoder + Attention\n";
	std::cout << "// --------------------------------------------------\n";
	EncoderDecoderAttn model = EncoderDecoderAttn(encoder, decoder_attn, 2, 2, 0.0);

	std::cout << "encdec(full_seq):\n" << model->forward(full_seq) << '\n';

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
	std::cout << "//Model Configuration & Training\n";
	std::cout << "// --------------------------------------------------\n";
	torch::manual_seed(23);

	encoder = Encoder(2, 2);
	decoder_attn = DecoderAttn(2, 2);
	model = EncoderDecoderAttn(encoder, decoder_attn, 2, 2, 0.5);
	model->to(device);

	auto loss_fnc = torch::nn::MSELoss();
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
			auto loss = loss_fnc(output, y);
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






