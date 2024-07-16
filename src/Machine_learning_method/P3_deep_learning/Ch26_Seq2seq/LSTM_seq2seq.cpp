/*
 * LSTM_seq2seq.cpp
 *
 *  Created on: Jul 9, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class S2SEncoderImpl : public torch::nn::Module {
public:
    /*由LSTM组成的序列到序列编码器。
    Args:
        inp_size: 嵌入层的输入维度
        embed_size: 嵌入层的输出维度
        num_hids: LSTM隐层向量维度
        num_layers: LSTM层数，本题目设置为4
    */
	S2SEncoderImpl() {}
	S2SEncoderImpl(int inp_size, int embed_size, int num_hids,
                 int num_layers, float dropout=0.) {

        embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(inp_size, embed_size));
        rnn = torch::nn::LSTM(torch::nn::LSTMOptions(embed_size, num_hids).num_layers(num_layers).dropout(dropout));
        register_module("embed", embed);
        register_module("rnn", rnn);
	}

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor inputs) {
        // inputs.shape(): (seq_length, embed_size)
        inputs = embed->forward(inputs);
        // output.shape(): (seq_length, num_hids)
        // states.shape(): (num_layers, num_hids)
		torch::Tensor output;
		std::tuple<torch::Tensor, torch::Tensor> state;
        std::tie(output, state) = rnn->forward(inputs);
        return std::make_tuple(output, state);
   }

private:
    torch::nn::Embedding embed{nullptr};
    torch::nn::LSTM rnn{nullptr};
};
TORCH_MODULE(S2SEncoder);

class S2SDecoderImpl : public torch::nn::Module {
public:
    /*由LSTM组成的序列到序列解码器。
    Args:
        inp_size: 嵌入层的输入维度。
        embed_size: 嵌入层的输出维度。
        num_hids: LSTM 隐层向量维度。
        num_layers: LSTM 层数，本题目设置为4。
    */
	S2SDecoderImpl() {}

	S2SDecoderImpl(int inp_size, int embed_size, int num_hids,
                 int _num_layers, float dropout=0.) {
        num_layers = _num_layers;
        embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(inp_size, embed_size));
        // 解码器 LSTM 的输入，由目标序列的嵌入向量和编码器的隐层向量拼接而成。
        rnn = torch::nn::LSTM(torch::nn::LSTMOptions(embed_size + num_hids, num_hids).num_layers(num_layers).dropout(dropout));
        linear = torch::nn::Linear(torch::nn::LinearOptions(num_hids, inp_size));
        register_module("embed", embed);
        register_module("rnn", rnn);
        register_module("linear", linear);
	}

    torch::Tensor init_state(std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> enc_outputs) {
    	std::tuple<torch::Tensor, torch::Tensor> t = std::get<1>(enc_outputs);
        return std::get<1>(t);
    }

    torch::Tensor forward(torch::Tensor inputs, torch::Tensor state) {
        // inputs.shape(): (seq_length, embed_size)
        inputs = embed->forward(inputs);

        // 广播 context，使其具有与 inputs 相同的长度
        // context.shape(): (seq_length, num_layers, embed_size)
        torch::Tensor context = state[-1].repeat({inputs.size(0), 1, 1});
        inputs = torch::cat({inputs, context}, 2);
        // output.shape(): (seq_length, num_hids)
        torch::Tensor output;
        std::tuple<torch::Tensor, torch::Tensor> _;
        std::tie(output, _) = rnn->forward(inputs);

        output = linear->forward(output);
        return output;
	}

private:
	int num_layers = 0;
    torch::nn::Embedding embed{nullptr};
    torch::nn::LSTM rnn{nullptr};
    torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(S2SDecoder);

class EncoderDecoder : public torch::nn::Module {
public:
    /*基于 LSTM 的序列到序列模型。
    Args:
        encoder: 编码器。
        decoder: 解码器。
    */
	EncoderDecoder(S2SEncoder _encoder, S2SDecoder _decoder) {
        encoder = _encoder;
        decoder = _decoder;
        register_module("encoder", encoder);
        register_module("decoder", decoder);
	}

	torch::Tensor forward(torch::Tensor enc_inp, torch::Tensor dec_inp) {
		std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> enc_out = encoder->forward(enc_inp);
		torch::Tensor dec_state = decoder->init_state(enc_out);

        return decoder->forward(dec_inp, dec_state);
	}
private:
    S2SEncoder encoder{nullptr};
    S2SDecoder decoder{nullptr};
};

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
/*
	torch::Tensor cx = torch::tensor({{-0.0471,  0.0795, -0.0969,  0.1363,  0.0578,  0.0570, -0.1311, -0.0112,
						         -0.0149, -0.3480,  0.0259, -0.1004,  0.1168, -0.1065,  0.0559,  0.1875},
						        {-0.0473,  0.0787, -0.0968,  0.1359,  0.0574,  0.0570, -0.1313, -0.0114,
						         -0.0153, -0.3476,  0.0259, -0.1004,  0.1163, -0.1062,  0.0558,  0.1871},
						        {-0.0474,  0.0777, -0.0971,  0.1360,  0.0568,  0.0568, -0.1317, -0.0113,
						         -0.0153, -0.3471,  0.0255, -0.1008,  0.1158, -0.1053,  0.0548,  0.1865},
						        {-0.0473,  0.0769, -0.0978,  0.1366,  0.0564,  0.0564, -0.1323, -0.0112,
						         -0.0149, -0.3466,  0.0248, -0.1015,  0.1154, -0.1039,  0.0531,  0.1860}});

	std::cout << cx.repeat({4, 1,1}) << '\n';
	std::cout << cx[-1] << '\n';
*/
    int inp_size = 10, embed_size = 8, num_hids = 16, num_layers = 4;
    S2SEncoder encoder = S2SEncoder(inp_size, embed_size, num_hids, num_layers);
    S2SDecoder decoder = S2SDecoder(inp_size, embed_size, num_hids, num_layers);
    EncoderDecoder model = EncoderDecoder(encoder, decoder);

    const std::string enc_inp_seq = "I love you";
    const std::string dec_inp_seq = "我 爱 你";

    // 自己构造的的词典
	std::map<std::string, torch::Tensor> word2vec;
	word2vec["I"] = torch::tensor({1, 0, 0, 0});
	word2vec["love"] = torch::tensor({0, 1, 0, 0});
	word2vec["you"] = torch::tensor({0, 0, 1, 0});
	word2vec["!"] = torch::tensor({0, 0, 0, 1});
	word2vec["我"] = torch::tensor({1, 0, 0, 0});
	word2vec["爱"] = torch::tensor({0, 1, 0, 0});
	word2vec["你"] = torch::tensor({0, 0, 1, 0});
	word2vec["！"] = torch::tensor({0, 0, 0, 1});

	std::vector<torch::Tensor> enc_inp, dec_inp;
	std::vector<std::string> enc_strs = stringSplit(enc_inp_seq,  ' ');
	std::vector<std::string> dec_strs = stringSplit(dec_inp_seq,  ' ');
	for(auto& str : enc_strs)
		enc_inp.push_back(word2vec[str].clone());

	 torch::Tensor enc_inp_T = torch::stack(enc_inp, 0);
	 std::cout << "enc_inp_T.shape: " << enc_inp_T.sizes() << '\n';

	for(auto& str : dec_strs)
		dec_inp.push_back(word2vec[str].clone());

	torch::Tensor dec_inp_T = torch::stack(dec_inp, 0);
	std::cout << "dec_inp_T.shape: " << dec_inp_T.sizes() << '\n';

	torch::Tensor model_out = model.forward(enc_inp_T, dec_inp_T);
	std::cout << "model_out:\n" << model_out << '\n';

	std::cout << "Done!\n";
	return 0;
}



