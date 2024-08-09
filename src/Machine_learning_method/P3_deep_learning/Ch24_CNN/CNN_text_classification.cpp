/*
 * CNN_text_classification.cpp
 *
 *  Created on: May 19, 2024
 *      Author: jiamny
 */

#include <torch/script.h> // One-stop header.
#include <torch/csrc/api/include/torch/utils.h>
#include <iostream>
#include <memory>
#include <regex>
#include <set>
#include <bits/stdc++.h>
#include <unistd.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace F = torch::nn::functional;

torch::Tensor prepare_sequence(std::vector<std::string> seq, std::map<std::string, int> to_index) {
    std::vector<int> idxs; // = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    for(auto&str : seq ) {
    	if( to_index.find(str) != to_index.end() ) {
    		idxs.push_back(to_index[str]);
    	} else {
    		idxs.push_back(to_index["<UNK>"]);
    	}
    }
    return torch::from_blob(idxs.data(), {static_cast<int>(idxs.size())}, at::TensorOptions(torch::kInt32)).clone();
}

class  CNNClassifier : public torch::nn::Module {
public:
    //kernel_sizes=(3, 4, 5)
	CNNClassifier(int vocab_size, int embedding_dim, int output_size, int kernel_dim=100,
			std::vector<int> kernel_sizes={3, 3, 3}, float drp=0.5) {

        //self.embedding = nn.Embedding(vocab_size, embedding_dim)
        embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim));
        register_module("embed", embed);

        //self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        convs = torch::nn::ModuleList();
        int i = 0;
        for(auto& K : kernel_sizes) {
        	torch::nn::Conv2d cov = torch::nn::Conv2d(
        			torch::nn::Conv2dOptions(1, kernel_dim, {K, embedding_dim}));
        	register_module("cov"+std::to_string(i), cov);
        	convs->push_back(cov);
        	i++;
        }
        // kernal_size = (K,D)
        droupout = torch::nn::Dropout(drp);
        //fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
		fc = torch::nn::Linear(torch::nn::LinearOptions(kernel_sizes.size() * kernel_dim ,  output_size));
		register_module("fc", fc);
	}

	torch::Tensor forward(torch::Tensor inputs, bool is_training=false) {
        inputs = embed->forward(inputs).unsqueeze(1); // (B,1,T,D)
        std::vector<torch::Tensor> conved;
		for(auto& it : convs->modules(false) ) {
			auto cov = dynamic_cast<torch::nn::Conv2dImpl*>(it.get());
			torch::Tensor cov_inputs = F::relu(cov->forward(inputs)).squeeze(3);
		    // 经过激活函数
		    conved.push_back(cov_inputs);
		}

		std::vector<torch::Tensor> max_conved;
		for(int i = 0; i < conved.size(); i++) {
			torch::Tensor cv = conved[i];
			max_conved.push_back(F::max_pool1d(cv, F::MaxPool1dFuncOptions(cv.size(2))).squeeze(2));
		}

		torch::Tensor concated = torch::cat(max_conved, 1);

        if( is_training )
            concated = droupout->forward(concated); // # (N,len(Ks)*Co)

        torch::Tensor out = fc->forward(concated);
        return F::log_softmax(out, 1);
	}

private:
	torch::nn::Embedding embed{nullptr};
	torch::nn::ModuleList convs;
	torch::nn::Dropout droupout{nullptr};
	torch::nn::Linear fc{nullptr};
};

std::tuple<torch::Tensor, torch::Tensor> pad_to_batch(std::map<std::string, int> word2index, std::vector<int> batch,
											std::vector<std::pair<torch::Tensor, torch::Tensor>> dt_p) {
	std::vector<torch::Tensor> x, y;
	int max_x = 0;
    for(auto& i : batch)	 {
    	//std::cout << "dt_p[i].first: " << dt_p[i].first.sizes() << " dt_p[i].second: " << dt_p[i].second.sizes() << '\n';
    	int c = dt_p[i].first.size(1);
    	if( c > max_x )
    		max_x = c;
    	x.push_back(dt_p[i].first);
    	y.push_back(dt_p[i].second);
    }

    //max_x = max([s.size(1) for s in x])
    //std::cout << "max_x: " << max_x << '\n';
    std::vector<torch::Tensor> x_dt;
    for( auto& xi : x ) {
    	if( xi.size(1) < max_x ) {
    		std::vector<int> idxs;
    		for(int i = 0; i < (max_x - xi.size(1)); i++)
    			idxs.push_back(word2index["<PAD>"]);

    		torch::Tensor pad = torch::from_blob(idxs.data(), {static_cast<int>(idxs.size())},
    												at::TensorOptions(torch::kInt32)).view({1, -1});
    		x_dt.push_back(torch::cat({xi, pad}, 1));
    	} else {
    		x_dt.push_back(xi);
    	}
    }
    return std::make_tuple(torch::cat(x_dt), torch::cat(y).view(-1));
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "load data\n";
	std::cout << "// -----------------------------------------------------------------\n";

	std::string file_name = "./data/train_5500.label.txt";

	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<std::vector<std::string>> X;
	std::vector<std::string> y;

	if( fL.is_open() ) {
		std::getline(fL, line);

		while ( std::getline(fL, line) ) {
			std::vector<std::string> strs = stringSplit(line, '\t');
			std::string s = strs[1].substr(0, strs[1].length() -1);
			s = std::regex_replace(s, std::regex("\\d"), std::string("#"));
			std::vector<std::string> x = stringSplit(s, ' ');
			X.push_back(x);
			y.push_back(strs[0]);
		}
	}
	fL.close();

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Build Vocab\n";
	std::cout << "// -----------------------------------------------------------------\n";

	std::set<std::string> vocab;
	int cnt = 0;
	for(int i = 0; i < X.size(); i++) {
		std::vector<std::string> x = X[i];
		for(auto& str : x ) {
			if( vocab.find(str) == vocab.end() ) {
				vocab.insert(str);
			}
			cnt++;
		}
	}

	std::cout << "tot = "  << cnt <<  " " << vocab.size() << '\n';

	std::map<std::string, int> word2index = {{"<PAD>", 0}, {"<UNK>", 1}};
	std::map<int, std::string> index2word = {{0, "<PAD>"}, {1, "<UNK>"}};
	for(auto& vo : vocab) {
	    if( word2index.find(vo) == word2index.end() ) {
	    	index2word[word2index.size()] = vo;
	        word2index[vo] = word2index.size();
	    }
	}

	std::map<std::string, int> target2index;
	std::map<int, std::string> index2target;

	for(auto& cl : y) {
	    if( target2index.find(cl) == target2index.end() ) {
	    	index2target[target2index.size()] = cl;
	        target2index[cl] = target2index.size();
	    }
	}
	std::cout << "word2index: "  << word2index.size() <<  " target2index: " << target2index.size() << '\n';

	std::vector<std::pair<torch::Tensor, torch::Tensor>> data_p;

	for( int i = 0; i < X.size(); i++ ) {
	    torch::Tensor d = prepare_sequence(X[i], word2index).view({1, -1});
	    int y_id = target2index[y[i]];
	    torch::Tensor c = torch::tensor({y_id}, torch::kInt64).view({1, -1});
	    data_p.push_back(std::make_pair(d, c));
	}

	std::cout << "data_p: " << data_p.size() << '\n';

	std::random_shuffle(data_p.begin(), data_p.end());

	int train_size = static_cast<int>(data_p.size() * 0.9);

	std::vector<std::pair<torch::Tensor, torch::Tensor>> train_p, test_p;

	for( int i = 0; i < data_p.size(); i++ ) {
		if( i < train_size ) {
			train_p.push_back(data_p[i]);
		} else {
			test_p.push_back(data_p[i]);
		}
	}
	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Training model \n";
	std::cout << "// -----------------------------------------------------------------\n";

	int EPOCH = 10;
	int BATCH_SIZE = 50;
	std::vector<int> KERNEL_SIZES = {3,3,3};
	int KERNEL_DIM = 100;
	float LR = 0.001;

	CNNClassifier model = CNNClassifier(word2index.size(), 300, target2index.size(), KERNEL_DIM, KERNEL_SIZES);
	model.to(device);
	model.train();

	auto loss_function = torch::nn::CrossEntropyLoss();
	auto optimizer = torch::optim::Adam(model.parameters(), LR);
	for(auto& epoch : range(EPOCH, 0)) {
		std::list<std::vector<int>> batchs = data_index_iter(train_p.size(), BATCH_SIZE, true);

		double losses = 0;
		int cnt = 0;
		for(auto& batch : batchs) {
			torch::Tensor inputs, targets, preds;
			std::tie(inputs,targets) = pad_to_batch(word2index, batch, train_p);

	        model.zero_grad();
	        preds = model.forward(inputs.to(device), true);

	        auto loss = loss_function(preds, targets.to(device));
	        losses += loss.data().item<float>();
	        cnt++;
	        loss.backward();
	        optimizer.step();
		}
		printf("[%d/%d] mean_loss : %0.4f\n", (epoch + 1), EPOCH, (losses/cnt));
	}

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Testing\n";
	std::cout << "// -----------------------------------------------------------------\n";
	model.eval();
	int accuracy = 0;

	for( int i = 0; i < test_p.size(); i++ ) {
		torch::Tensor test = test_p[i].first.to(device);
		torch::Tensor _, pred;
		std::tie(_, pred) = torch::max(model.forward(test, false), 1);
	    int pd = pred.cpu().data().item<long>();
	    int target = test_p[i].second.squeeze().data().item<long>();
	    if( pd == target )
	        accuracy += 1;
	}
	printf("accuracy = %0.2f\n", (accuracy * 1.0/test_p.size()) * 100);

	std::cout << "Done!\n";
	return 0;
}

