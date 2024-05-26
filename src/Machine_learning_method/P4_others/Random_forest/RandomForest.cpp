/*
 * RandomForest.cpp
 *
 *  Created on: May 26, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <vector>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

struct DecisionTree : public torch::nn::Module {
	torch::nn::Linear fc{nullptr};

	DecisionTree(int input, int output) {
		fc = torch::nn::Linear(torch::nn::LinearOptions(input, output));
		register_module("fc", fc);
	}

	torch::Tensor forward(torch::Tensor x) {
		return fc->forward(x);
	}
};

struct RandomForest : public torch::nn::Module {
	torch::nn::ModuleList tree;

	RandomForest(int num_tree, int input, int output) {

		for(auto& i : range(num_tree, 0)) {
			tree->push_back(torch::nn::Linear(torch::nn::LinearOptions(input, output)));
		}
		register_module("tree", tree);
	}

	torch::Tensor forward(torch::Tensor x) {
		std::vector<torch::Tensor> fwd;

		for(const auto& md : *tree ) {
			torch::Tensor f = md->as<torch::nn::Linear>()->forward(x);
			//std::optional<long int> dim = {1};
			//fwd.push_back(torch::argmax(f, dim).clone());
			fwd.push_back(f);
		}
		torch::Tensor tt = torch::stack(fwd);
		c10::OptionalArrayRef<long int> dim = {0};

		/*
		torch::Tensor r = torch::zeros({x.size(0)});

		for(int i = 0; i < x.size(0); i++) {
			auto m = torch::_unique2(t.index({Slice(), i}), true, true, true);
			r.index_put_({i}, torch::argmax(std::get<2>(m)));
		}
		*/
		return tt.mean(dim);
	}
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	int input = 10, output = 2, num_tree = 10, num_sample = 10;

	torch::Tensor x_train = torch::randn({num_sample, input});
	torch::Tensor y_train = torch::randint(2, {num_sample});

	std::cout << y_train.dtype() << '\n';

	auto model = RandomForest(num_tree, input, output);

	auto optimizer = torch::optim::SGD(model.parameters(), 0.1);
	auto criterion = torch::nn::CrossEntropyLoss();

	int num_epoch = 10;

	for(auto& epoch : range(num_epoch, 0)) {
		optimizer.zero_grad();
		auto out = model.forward(x_train);
		auto loss = criterion(out, y_train);
		loss.backward();
		optimizer.step();
		std::cout << "loss:\n" << loss << '\n';
		std::optional<long int> dim = {1};
		auto y_pred = torch::argmax(out, dim);
		auto correct = torch::sum((y_pred==y_train).to(torch::kInt32)).data().item<int>();
		std::cout << "y_pred:\n" << y_pred << '\n' << correct << '\n';
	}

	torch::Tensor x_test = torch::randn({10, input});
	torch::Tensor prediction = model.forward(x_test);
	std::cout << "prediction:\n" << prediction << '\n';

	std::cout << "Done!\n";
	return 0;
}


