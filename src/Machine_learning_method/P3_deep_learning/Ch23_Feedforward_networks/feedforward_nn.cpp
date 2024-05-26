/*
 * feedforward_nn.cpp
 *
 *  Created on: May 11, 2024
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
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/activation.h"
#include "../../../Utils/loss.h"

using torch::indexing::Slice;
using torch::indexing::None;

torch::Tensor dnn_xnor(torch::Tensor X, torch::Tensor W1, torch::Tensor b1,
		torch::Tensor W2, torch::Tensor b2) {
	torch::Tensor Z1 = W1.t().mm(X) + b1;
	torch::Tensor H = Sigmoid(Z1);

	torch::Tensor Z2 = W2.t().mm(H) + b2;
	torch::Tensor Y = Sigmoid(Z2);

    return Y;
}

torch::Tensor accuracy_score(torch::Tensor y, torch::Tensor p) {
	c10::OptionalArrayRef<long int> dim = {0};
	torch::Tensor accuracy = torch::sum(y == p, dim) / y.size(0);
    return accuracy;
}

torch::Tensor normalization(torch::Tensor X) {
    /*
    :param X: Input tensor
    :return: Normalized input using l2 norm.
    */
	const std::optional<c10::Scalar> p = {2};
	c10::ArrayRef<long int> dim = {-1};
	torch::Tensor l2 = torch::norm(X, p, dim);
    l2.masked_fill_(l2 == 0, 1);
    return X / l2.unsqueeze(1);
}


class MultiLayerPerceptron {
public:
	MultiLayerPerceptron(int _n_hidden, int _n_iterations=1000, double _learning_rate=0.001) {
        n_hidden = _n_hidden;
        n_iterations = _n_iterations;
        learning_rate = _learning_rate;
        loss = _CrossEntropy();
	}

	void initalize_weight(torch::Tensor X, torch::Tensor y) {
        int n_samples  = X.size(0);
        int n_features = X.size(1);
        int n_outputs = y.size(1);
        double limit = (1. / torch::sqrt(torch::tensor({n_features}, torch::kDouble))).data().item<double>();
        W = torch::empty({n_features, n_hidden}).uniform_(-limit, limit).to(torch::kDouble).to(X.device());

        W0 = torch::zeros({1, n_hidden}, torch::kDouble).to(X.device());
        limit = (1. / torch::sqrt(torch::tensor(n_hidden))).data().item<double>();

        V = torch::empty({n_hidden, n_outputs}).uniform_(-limit, limit).to(torch::kDouble).to(X.device());
        V0 = torch::zeros({1, n_outputs}, torch::kDouble).to(X.device());
	}

    void fit(torch::Tensor X, torch::Tensor y) {
        initalize_weight(X, y);
        for(auto& i : range(n_iterations, 0) ) {
        	torch::Tensor hidden_input =  torch::mm(X, W) + W0;
        	torch::Tensor hidden_output = hidden_activation.forward(hidden_input);

        	torch::Tensor output_layer_input = torch::mm(hidden_output, V) + V0;
        	torch::Tensor y_pred  = output_activation.forward(output_layer_input);

        	torch::Tensor grad_wrt_first_output = loss.gradient(y,
        											y_pred) * output_activation.gradient(output_layer_input);

        	torch::Tensor grad_v = torch::mm(hidden_output.t(), grad_wrt_first_output);
        	c10::OptionalArrayRef<long int> dim = {0};
        	torch::Tensor grad_v0 = torch::sum(grad_wrt_first_output, dim, true);

        	torch::Tensor grad_wrt_first_hidden = torch::mm(grad_wrt_first_output,
        											V.t()) * hidden_activation.gradient(hidden_input);
        	torch::Tensor grad_w = torch::mm(X.t(), grad_wrt_first_hidden);
			c10::OptionalArrayRef<long int> dm = {0};
			torch::Tensor grad_w0 = torch::sum(grad_wrt_first_hidden, dm, true);

			// Update weights (by gradient descent)
            // Move against the gradient to minimize loss
			//torch::NoGradGuard no_grad;
            V -= learning_rate * grad_v;
            V0 -= learning_rate * grad_v0;
            W -= learning_rate * grad_w;
            W0 -= learning_rate * grad_w0;
        }
    }

    // Use the trained model to predict labels of X
    torch::Tensor predict(torch::Tensor X) {
        // Forward pass:
    	torch::Tensor hidden_input = torch::mm(X, W) + W0;
    	torch::Tensor hidden_output = hidden_activation.forward(hidden_input);
    	torch::Tensor output_layer_input = torch::mm(hidden_output, V) + V0;
    	torch::Tensor y_pred = output_activation.forward(output_layer_input);
        return y_pred;
    }

private:
	int n_hidden, n_iterations;
	float learning_rate;
	_Sigmoid hidden_activation;
	_Softmax output_activation;
	_CrossEntropy loss;
	 torch::Tensor W, W0, V, V0;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Running on GPU." : "Running on CPU.") << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  XNOR\n";
	std::cout << "// --------------------------------------------------\n";

	// 定义网络的权重W和偏置b
	torch::Tensor W1 = torch::tensor({{20, -20}, {20, -20}}, torch::kFloat32).to(device);
	torch::Tensor b1 = torch::tensor({{-30}, {10}}, torch::kFloat32).to(device);
	torch::Tensor W2 = torch::tensor({{20}, {20}}, torch::kFloat32).to(device);
	torch::Tensor b2 = torch::tensor({{-10}}, torch::kFloat32).to(device);
	torch::Tensor X = torch::tensor({{0, 0, 1, 1},
	              {0, 1, 0, 1}}, torch::kFloat32).to(device);

	std::cout << "dnn_xnor:\n " << dnn_xnor(X, W1, b1, W2, b2) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  多层感知机, MultiLayerPerceptron\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor labels = torch::tensor({1, 2, 3, 4, 5, 1}, torch::kLong);
	torch::Tensor ct = to_categorical(labels);
	//std::cout << "ct:\n " << ct << '\n';

	std::ifstream file;
	std::string path = "./data/dataset_28_optdigits.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(num_records, true);

	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));
	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> iMap;
	std::cout << "iMap.empty(): " << iMap.empty() << '\n';

	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, false, true);

    file.close();

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	std::cout << "test_lab:\n";
	printVector(tensorTovector(test_lab.to(torch::kDouble)));

	train_dt = train_dt.to(torch::kDouble).to(device);
	test_dt = test_dt.to(torch::kDouble).to(device);
	train_lab = to_categorical(train_lab.squeeze().to(torch::kLong)).to(device);
	test_lab = to_categorical(test_lab.squeeze().to(torch::kLong)).to(device);
	std::cout << "test_lab[0:10]:\n" << test_lab.index({Slice(0,10), Slice()}) << '\n';

	int n_hidden = 16;
	int n_iterations=1000;
	double learning_rate=0.001;

	// MLP
	MultiLayerPerceptron clf = MultiLayerPerceptron(n_hidden, n_iterations, learning_rate);
    clf.fit(train_dt, train_lab);

    std::optional<long int> dim = {1};
    torch::Tensor y_pred = torch::argmax(clf.predict(test_dt), dim);
    std::optional<long int> dm = {1};
    torch::Tensor y_test = torch::argmax(test_lab, dm);
    std::cout << "y_pred:\n";
    printVector(tensorTovector(y_pred.to(torch::kDouble).to(torch::kCPU)));
    std::cout << "y_test:\n";
    printVector(tensorTovector(y_test.to(torch::kDouble).to(torch::kCPU)));

    torch::Tensor accuracy = accuracy_score(y_test, y_pred);
    printf("Accuracy: %3.1f\n", accuracy.data().item<double>() * 100);

	std::cout << "Done!\n";
	return 0;
}




