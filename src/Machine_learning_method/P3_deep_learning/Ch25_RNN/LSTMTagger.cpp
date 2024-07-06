/*
 * LSTMTagger.cpp
 *
 *  Created on: Jul 6, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <map>
#include <iterator>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;


torch::Tensor prepare_sequence(std::vector<std::string> seq, std::map<std::string, int> to_ix) {
    std::vector<int> idxs;
    //= [to_ix[w] for w in seq]
    for(auto& w : seq ) {
    	idxs.push_back(to_ix[w]);
    }
    torch::Tensor tensor = torch::tensor(idxs).to(torch::kLong);
    return tensor;
}

class LSTMTagger : public torch::nn::Module {
public:
	std::tuple<torch::Tensor, torch::Tensor> hidden;

	LSTMTagger(int embedding_dim, int _hidden_dim, int vocab_size, int tagset_size) {
        hidden_dim = _hidden_dim;
        word_embeddings = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim));
        lstm = torch::nn::LSTM(torch::nn::LSTMOptions(embedding_dim, hidden_dim));
        hidden2tag = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, tagset_size));
        hidden = init_hidden();
        register_module("word_embeddings", word_embeddings);
        register_module("hidden2tag", hidden2tag);
        register_module("lstm", lstm);
	}

	//初始化隐含状态State及C
	std::tuple<torch::Tensor, torch::Tensor> init_hidden() {
        return std::make_tuple(torch::zeros({1, 1, hidden_dim}),
                torch::zeros({1, 1, hidden_dim}));
	}

	torch::Tensor forward(torch::Tensor sentence) {
        //获得词嵌入矩阵embeds
		torch::Tensor embeds = word_embeddings->forward(sentence);
        //按lstm格式，修改embeds的形状
		torch::Tensor lstm_out;
        std::tie(lstm_out, hidden) = lstm->forward(embeds.view({sentence.size(0), 1, -1}), hidden);
        //修改隐含状态的形状，作为全连接层的输入
		torch::Tensor tag_space = hidden2tag->forward(lstm_out.view({sentence.size(0), -1}));

        //计算每个单词属于各词性的概率
		torch::Tensor tag_scores = torch::log_softmax(tag_space,1);
        return tag_scores;
	}

private:
	int hidden_dim = 0, embedding_dim = 0;
	torch::nn::Embedding word_embeddings{nullptr};
	torch::nn::LSTM lstm{nullptr};
	torch::nn::Linear hidden2tag{nullptr};
};


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// 定义训练数据

	std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> training_data = {
			std::make_pair(stringSplit("The cat ate the fish", ' '), std::vector<std::string>{"DET", "NN", "V", "DET", "NN"}),
			std::make_pair(stringSplit("They read that book", ' '), std::vector<std::string>{"NN", "V", "DET", "NN"})};

	// 定义测试数据
	std::vector<std::string> testing_data = stringSplit("They ate the fish", ' ');
	for(auto& s : testing_data)
		std::cout << s << " ";
	std::cout << '\n';

	std::map<std::string, int> word_to_ix; // 单词的索引字典
	for(auto& t : training_data) {
		std::vector<std::string> sent = t.first;
		std::vector<std::string> tags = t.second;
	    for(auto& word : sent ) {
	        if( word_to_ix.count(word) ==  0 ) {
	            word_to_ix[word] = word_to_ix.size();
	        }
	    }
	}
	std::map<std::string, int>::iterator it = word_to_ix.begin();
	while( it != word_to_ix.end() ) {
		std::cout << it->first << " " << it->second << "\t";
		it++;
	}
	std::cout << '\n';

	std::map<std::string, int> tag_to_ix = {
			std::pair<std::string, int>("DET", 0),
			std::pair<std::string, int>("NN", 1),
			std::pair<std::string, int>("V", 2)}; // 手工设定词性标签数据字典

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 训练LSTMTagger网络\n";
	std::cout << "// --------------------------------------------------\n";

	int EMBEDDING_DIM=10;
	int HIDDEN_DIM=3;  //这里等于词性个数

	LSTMTagger model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, word_to_ix.size(), tag_to_ix.size());
	auto loss_function = torch::nn::NLLLoss();
	auto optimizer = torch::optim::SGD(model.parameters(), 0.1);

	torch::Tensor inputs = prepare_sequence(training_data[0].first, word_to_ix);
	torch::Tensor tag_scores = model.forward(inputs);
	for(auto& s : training_data[0].first)
		std::cout << s << " ";
	std::cout << '\n';
	std::cout << "inputs:\n" << inputs << '\n';
	std::cout << "tag_scores:\n" << tag_scores << '\n';
	std::cout << "torch::max(tag_scores,1):\n" << std::get<0>(torch::max(tag_scores,1)) << '\n';

	for(auto& epoch : range(400, 0) ) {
	    for(auto& t : training_data ) {
	    	std::vector<std::string> sentence = std::get<0>(t), tags  = std::get<1>(t);
	        model.zero_grad();
	        model.hidden = model.init_hidden();
	        // 按网络要求的格式处理输入数据和真实标签数据
	        torch::Tensor sentence_in = prepare_sequence(sentence, word_to_ix);
	        torch::Tensor targets = prepare_sequence(tags, tag_to_ix);

	        tag_scores = model.forward(sentence_in);
			// 计算损失，反向传递梯度及更新模型参数
	        auto loss = loss_function(tag_scores, targets);
	        loss.backward();
	        optimizer.step();
	    }
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 查看模型训练的结果\n";
	std::cout << "// --------------------------------------------------\n";
	inputs = prepare_sequence(training_data[0].first, word_to_ix);
	tag_scores = model.forward(inputs);
	for(auto& s : training_data[0].first)
		std::cout << s << " ";
	std::cout << '\n';
	std::cout << "inputs:\n" << inputs << '\n';
	std::cout << "tag_scores:\n" << tag_scores << '\n';
	std::cout << "torch::max(tag_scores,1):\n" << std::get<0>(torch::max(tag_scores,1)) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 测试模型\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor test_inputs = prepare_sequence(testing_data, word_to_ix);
	torch::Tensor tag_scores01 = model.forward(test_inputs);
	for(auto& s : testing_data)
		std::cout << s << " ";
	std::cout << '\n';
	std::cout << "test_inputs:\n" << test_inputs << '\n';
	std::cout << "tag_scores01:\n" << tag_scores01 << '\n';
	std::cout << "torch::max(tag_scores01,1):\n" << std::get<0>(torch::max(tag_scores01,1)) << '\n';

	std::cout << "Done!\n";
	return 0;
}



