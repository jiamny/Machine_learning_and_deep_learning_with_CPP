/*
 * Encoder_decoder.h
 *
 *  Created on: Jul 18, 2024
 *      Author: jiamny
 */

#ifndef ENCODER_DECODER_H_
#define ENCODER_DECODER_H_
#pragma once

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

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

#endif /* ENCODER_DECODER_H_ */
