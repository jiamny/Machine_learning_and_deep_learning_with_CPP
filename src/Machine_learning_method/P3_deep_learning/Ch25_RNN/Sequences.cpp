/*
 * Sequences.cpp
 *
 *  Created on: Jul 2, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class GRUCell : public torch::nn::Module {
public:
	GRUCell(int _input_size, int _hidden_size, int _output_size) {
		input_size = _input_size;
        hidden_size = _hidden_size;
        output_size = _output_size;
        gate = torch::nn::Linear(torch::nn::LinearOptions(input_size + hidden_size, hidden_size));
        output = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_size));
        sigmoid = torch::nn::Sigmoid();
        tanh = torch::nn::Tanh();
        softmax = torch::nn::LogSoftmax(1);
        register_module("gate", gate);
        register_module("output", output);
        register_module("sigmoid", sigmoid);
        register_module("tanh", tanh);
        register_module("softmax", softmax);
	}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor hidden) {
    	torch::Tensor combined = torch::cat({input, hidden}, 1);
    	torch::Tensor z_gate = sigmoid->forward(gate->forward(combined));
    	torch::Tensor r_gate = sigmoid->forward(gate->forward(combined));

    	torch::Tensor combined01 = torch::cat({input, torch::mul(hidden, r_gate)}, 1);
    	torch::Tensor h1_state = tanh->forward(gate->forward(combined01));

    	torch::Tensor h_state = torch::add(torch::mul((1-z_gate), hidden), torch::mul(h1_state, z_gate));
    	torch::Tensor ot = output->forward(h_state);
        ot = softmax->forward(ot);
        return std::make_tuple(ot, h_state);
    }

    torch::Tensor initHidden(void) {
        return torch::zeros({1, hidden_size});
    }
private:
	int input_size = 0, hidden_size = 0, output_size = 0;
	torch::nn::LogSoftmax softmax{nullptr};
	torch::nn::Tanh tanh{nullptr};
	torch::nn::Sigmoid sigmoid{nullptr};
	torch::nn::Linear gate{nullptr}, output{nullptr};

};

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Tensor X = torch::tensor({1., 2.}).unsqueeze(0);;
	torch::Tensor state = torch::tensor({0.0, 0.0}).unsqueeze(0);;
	torch::Tensor w_cell_state = torch::tensor({{0.1, 0.2}, {0.3, 0.4},{0.5, 0.6}});
	torch::Tensor b_cell = torch::tensor({0.1, -0.1}).unsqueeze(0);;
	torch::Tensor w_output = torch::tensor({{1.0}, {2.0}});
	float b_output = 0.1;

	for(auto& i : range(static_cast<int>(X.size(1)), 0)) {
	    state = torch::cat({state, X.index({0,i}).reshape({1,1})}, 1);
	    torch::Tensor before_activation = torch::mm(state, w_cell_state) + b_cell;
	    state = torch::tanh(before_activation);
	    torch::Tensor final_output = torch::mm(state, w_output) + b_output;
	    std::cout <<"--------- 状态值_" << i << ":\n" << state << '\n';
	    std::cout <<"========= 输出值_" << i << ":\n" << final_output << '\n';
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// RNN\n";
	std::cout << "// --------------------------------------------------\n";
	torch::nn::RNN rnn = torch::nn::RNN(torch::nn::RNNOptions(10, 20).num_layers(2));
	auto states = rnn->named_parameters();
	std::cout << "wih形状: " << states["weight_ih_l0"].sizes() << '\n';
	std::cout << "whh形状: " << states["weight_hh_l0"].sizes() << '\n';
	std::cout << "bih形状: " << states["bias_hh_l0"].sizes() << '\n';

	torch::Tensor input = torch::randn({100,32,10});
	torch::Tensor h_0 = torch::randn({2,32,20});
	torch::Tensor output,h_n;
	std::tie(output,h_n) = rnn->forward(input,h_0);
	std::cout << "RNN output: " << output.sizes() << '\n';
	std::cout << "RNN h_n: " << h_n.sizes() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// RNNCell\n";
	std::cout << "// --------------------------------------------------\n";
	torch::nn::RNNCell rnn_cell = torch::nn::RNNCell(torch::nn::RNNCellOptions(10, 20));
	torch::Tensor input_cell = torch::randn({100,32,10});
	torch::Tensor h_cell_0 = torch::randn({2, 32,20});
	std::tie(output,h_n) = rnn->forward(input_cell, h_cell_0);
	std::cout << "RNNCell output: " << output.sizes() << '\n';
	std::cout << "RNNCell h_n: " << h_n.sizes() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// GRUCell\n";
	std::cout << "// --------------------------------------------------\n";
	GRUCell grucell = GRUCell(10, 20, 10);

	input = torch::randn({32,10});
	h_0 = torch::randn({32,20});

	torch::Tensor hn;
	std::tie(output, hn) = grucell.forward(input,h_0);
	std::cout << "GRUCell output: " << output.sizes() << '\n';
	std::cout << "GRUCell hn: " << hn.sizes() << '\n';

	torch::nn::GRUCell gru_cell = torch::nn::GRUCell(torch::nn::GRUCellOptions(10, 20));
	output = gru_cell->forward(input);
	std::cout << "torch GRUCell output: " << output.sizes() << '\n';

	std::cout << "Done!\n";
	return 0;
}
