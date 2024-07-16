/*
 * CNN_seq2seq.cpp
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

namespace F = torch::nn::functional;

class CNNEncoderImpl : public torch::nn::Module {
public:
    /*序列到序列 CNN 编码器。
    Args:
        inp_dim: 嵌入层的输入维度。
        emb_dim: 嵌入层的输出维度。
        hid_dim: CNN 隐层向量维度。
        num_layers: CNN 层数。
        kerner_size: 卷积核大小。
    */
	CNNEncoderImpl() {}
	CNNEncoderImpl(int inp_dim, int emb_dim, int hid_dim,
                 int num_layers, int kernel_size) {
        embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(inp_dim, emb_dim));
        register_module("embed", embed);
        emb2hid = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, static_cast<int>(hid_dim / 2)));
        register_module("emb2hid", emb2hid);
        hid2emb = torch::nn::Linear(torch::nn::LinearOptions(static_cast<int>(hid_dim / 2), emb_dim));
        register_module("hid2emb", hid2emb);

        convs = torch::nn::ModuleList();
        for(auto& i : range(num_layers, 0)) {
        	torch::nn::Conv1d cov = torch::nn::Conv1d(
        			torch::nn::Conv1dOptions(emb_dim, hid_dim, kernel_size).padding(static_cast<int>((kernel_size - 1)/2)));
        	register_module("cov"+std::to_string(i), cov);
        	convs->push_back(cov);
        }
	}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inputs) {
        // inputs.shape(): (src_len, inp_dim)
        // conv_inp.shape(): (src_len, emb_dim)
    	torch::Tensor conv_inp = embed->forward(inputs).permute({0, 2, 1});

    	torch::Tensor conved;
        for(auto& it : convs->modules(false) ) {
        	auto cov = dynamic_cast<torch::nn::Conv1dImpl*>(it.get());
            // 进行卷积运算
            // conv_out.shape(): (src_len, hid_dim)
        	torch::Tensor conv_out = cov->forward(conv_inp);
            // 经过激活函数
        	conved = F::glu(conv_out, 1);
            // 残差连接运算
            conved = hid2emb->forward(conved.permute({0, 2, 1})).permute({0, 2, 1});
            conved = conved + conv_inp;
            conv_inp = conved.clone();
        }

        // 卷积输出与词嵌入 element-wise 点加进行注意力运算
        // combined.shape(): (src_len, emb_dim)
        torch::Tensor combined = conved + conv_inp;

        return std::make_tuple(conved, combined);
    }

private:
	torch::nn::Embedding embed{nullptr};
	torch::nn::Linear emb2hid{nullptr}, hid2emb{nullptr};
	//std::vector<torch::nn::Conv1d> convs;
	torch::nn::ModuleList convs;
};
TORCH_MODULE(CNNEncoder);

class CNNDecoderImpl : public torch::nn::Module {
public:
    /*序列到序列 CNN 解码器。
    Args:
        out_dim: 嵌入层的输入维度。
        emb_dim: 嵌入层的输出维度。
        hid_dim: CNN 隐层向量维度。
        num_layers: CNN 层数。
        kernel_size: 卷积核大小。
    */
	CNNDecoderImpl() {};

	CNNDecoderImpl(int out_dim, int emb_dim, int hid_dim,
                 int num_layers, int _kernel_size, int _trg_pad_idx) {
        kernel_size = _kernel_size;
        trg_pad_idx = _trg_pad_idx;

        embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(out_dim, emb_dim));
        register_module("embed", embed);
        emb2hid = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, static_cast<int>(hid_dim / 2)));
		register_module("emb2hid", emb2hid);
        hid2emb = torch::nn::Linear(torch::nn::LinearOptions(static_cast<int>(hid_dim / 2), emb_dim));
		register_module("hid2emb", hid2emb);
        attn_hid2emb = torch::nn::Linear(torch::nn::LinearOptions(static_cast<int>(hid_dim / 2), emb_dim));
        register_module("attn_hid2emb", attn_hid2emb);
        attn_emb2hid = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, static_cast<int>(hid_dim / 2)));
        register_module("attn_emb2hid", attn_emb2hid);
        fc_out = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, out_dim));
		register_module("fc_out", fc_out);

		convs = torch::nn::ModuleList();
		for(auto& i : range(num_layers, 0)) {
			torch::nn::Conv1d cov = torch::nn::Conv1d(
			        			torch::nn::Conv1dOptions(emb_dim, hid_dim, kernel_size));
			register_module("cov"+std::to_string(i), cov);
			convs->push_back(cov);
		}
	}

    std::tuple<torch::Tensor, torch::Tensor> calculate_attention(torch::Tensor embed,
    		torch::Tensor conved, torch::Tensor encoder_conved, torch::Tensor encoder_combined) {
        // embed.shape(): (trg_len, emb_dim)
        // conved.shape(): (hid_dim, trg_len)
        // encoder_conved.shape(), encoder_combined.shape(): (src_len, emb_dim)
        // 进行注意力层第一次线性运算调整维度
    	torch::Tensor conved_emb = attn_hid2emb->forward(conved.permute({0, 2, 1})).permute({0, 2, 1});

        // conved_emb.shape(): (trg_len, emb_dim])
    	torch::Tensor combined = conved_emb + embed;
        // print(combined.size(), encoder_conved.size())
    	torch::Tensor energy = torch::matmul(combined.permute({0, 2, 1}), encoder_conved);

        // attention.shape(): (trg_len, emb_dim])
    	torch::Tensor attention = F::softmax(energy, 2);
    	torch::Tensor attended_encoding = torch::matmul(attention, encoder_combined.permute({0, 2, 1}));

        // attended_encoding.shape(): (trg_len, emd_dim)
        // 进行注意力层第二次线性运算调整维度
        attended_encoding = attn_emb2hid->forward(attended_encoding);

        // attended_encoding.shape(): (trg_len, hid_dim)
        // 残差计算
        torch::Tensor attended_combined = conved + attended_encoding.permute({0, 2, 1});

        return std::make_tuple(attention, attended_combined);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor targets,
    												 torch::Tensor encoder_conved, torch::Tensor encoder_combined) {
        // targets.shape(): (trg_len, out_dim)
        // encoder_conved.shape(): (src_len, emb_dim)
        // encoder_combined.shape(): (src_len, emb_dim)
    	torch::Tensor conv_inp = embed->forward(targets).permute({0, 2, 1});

        int src_len = conv_inp.size(0);
        int hid_dim = conv_inp.size(1);

        torch::Tensor conved;
        torch::Tensor attention;

        for(auto& it : convs->modules(false) ) {
        	auto cov = dynamic_cast<torch::nn::Conv1dImpl*>(it.get());
            // need to pad so decoder can't "cheat"
        	torch::Tensor padding = torch::zeros({src_len, hid_dim, kernel_size - 1}).fill_(trg_pad_idx);

        	torch::Tensor padded_conv_input = torch::cat({padding, conv_inp}, -1);

            // padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            // 经过卷积运算
        	conved = cov->forward(padded_conv_input);
            // 经过激活函数
            conved = F::glu(conved, 1);
            // 注意力分数计算
            std::tie(attention, conved) = calculate_attention(conv_inp, conved,
                                                         encoder_conved,
                                                         encoder_combined);
            // 残差连接计算
            conved = hid2emb->forward(conved.permute({0, 2, 1})).permute({0, 2, 1});
            conved = conved + conv_inp;
            conv_inp = conved.clone();
        }

        torch::Tensor output = fc_out->forward(conved.permute({0, 2, 1}));
        return std::make_tuple(output, attention);
    }

private:
	int kernel_size = 0, trg_pad_idx = 0;
	torch::nn::Embedding embed{nullptr};
	torch::nn::Linear emb2hid{nullptr}, hid2emb{nullptr}, attn_hid2emb{nullptr}, attn_emb2hid{nullptr}, fc_out{nullptr};
	//std::vector<torch::nn::Conv1d> convs;
	torch::nn::ModuleList convs;
};
TORCH_MODULE(CNNDecoder);

class EncoderDecoder : public torch::nn::Module {
public:
    //序列到序列 CNN 模型。
	EncoderDecoder(CNNEncoder _encoder, CNNDecoder _decoder) {
        encoder = _encoder;
        decoder = _decoder;
        register_module("encoder", encoder);
        register_module("decoder", decoder);
	}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor enc_inp, torch::Tensor dec_inp) {
        // 编码器，将源句子编码为向量输入解码器进行解码。
    	torch::Tensor encoder_conved, encoder_combined;
        std::tie(encoder_conved, encoder_combined) = encoder->forward(enc_inp);

        // 解码器，根据编码器隐藏状态和解码器输入预测下一个单词的概率
        // 注意力层，源句子和目标句子之间进行注意力运算从而对齐
		torch::Tensor output, attention;
        std::tie(output, attention) = decoder->forward(dec_inp, encoder_conved, encoder_combined);

        return std::make_tuple(output, attention);
    }
private:
    CNNEncoder encoder{nullptr};
    CNNDecoder decoder{nullptr};
};

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// 构建一个基于CNN的序列到序列模型
	int inp_dim = 8, out_dim = 10, emb_dim = 12, hid_dim = 16, num_layers = 1, kernel_size = 3;

	CNNEncoder encoder = CNNEncoder(inp_dim, emb_dim, hid_dim, num_layers, kernel_size);
	CNNDecoder decoder = CNNDecoder(out_dim, emb_dim, hid_dim, num_layers, kernel_size, 0);
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

	std::tuple<torch::Tensor, torch::Tensor> model_out = model.forward(enc_inp_T, dec_inp_T);
	std::cout << "model_out.output:\n" << std::get<0>(model_out) << '\n'
			  << "model_out.attention:\n" << std::get<1>(model_out) << '\n';

	std::cout << "Done!\n";
	return 0;
}



