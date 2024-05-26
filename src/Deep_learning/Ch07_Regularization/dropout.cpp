#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../../Utils/fashion.h"
#include "../../Utils/helpfunction.h"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor dropout_layer(torch::Tensor X, double dropout) {
    assert( 0 <= dropout <= 1);

    //In this case, all elements are dropped out
    if( dropout == 1 )
        return torch::zeros_like(X).to(X.device());

    // In this case, all elements are kept
    if( dropout == 0 )
        return X;

    auto mask = (torch::rand(X.sizes(), dtype(torch::kDouble)) > dropout);

    return mask.to(X.device()) * X / (1.0 - dropout);
}

struct NetImpl : public torch::nn::Module {

public:
	int64_t num_inputs;
	NetImpl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1,
				int64_t num_hiddens2, bool is_training);

    torch::Tensor forward(torch::Tensor X);
private:
    bool training=false;
    torch::nn::Linear lin1{nullptr}, lin2{nullptr}, lin3{nullptr};
};

TORCH_MODULE(Net);

NetImpl::NetImpl(int64_t inputs, int64_t num_outputs, int64_t num_hiddens1,
							int64_t num_hiddens2, bool is_training) {
	training = is_training;
	num_inputs = inputs;
	lin1 = torch::nn::Linear(torch::nn::LinearOptions(num_inputs, num_hiddens1));
	lin2 = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens1, num_hiddens2));
	lin3 = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens2, num_outputs));
	register_module("fc1", lin1);
	register_module("fc2", lin2);
	register_module("fc3", lin3);
}

torch::Tensor NetImpl::forward(torch::Tensor x) {
	float dropout1 = 0.2, dropout2 = 0.5;

//	auto H1 = torch::nn::functional::relu(lin1->forward(x.view({x.size(0), -1})));
	auto H1 = torch::nn::functional::relu(lin1->forward(x.reshape({-1, num_inputs})));
//	auto H1 = torch::nn::functional::relu(lin1->forward(x));
	// Use dropout only when training the model
	if( training ) {
		// Add a dropout layer after the first fully connected layer
		H1 = dropout_layer(H1, dropout1);
	}

	auto H2 = torch::nn::functional::relu(lin2->forward(H1));

	if( training ) {
		// Add a dropout layer after the second fully connected layer
		H2 = dropout_layer(H2, dropout2);
	}
	auto out = lin3->forward(H2);
	return out;
}

struct NeuralNetImpl : public torch::nn::Module {
 public:
    NeuralNetImpl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1, int64_t num_hiddens2);

    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(NeuralNet);


NeuralNetImpl::NeuralNetImpl(int64_t num_inputs, int64_t num_outputs,
								int64_t num_hiddens1, int64_t num_hiddens2) :
	fc1(num_inputs, num_hiddens1), fc2(num_hiddens1, num_hiddens2), fc3(num_hiddens2, num_outputs) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
	float dropout1 = 0.2, dropout2 = 0.5;

    x = torch::nn::functional::relu(fc1->forward(x));
    x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout1));
    x = torch::nn::functional::relu(fc2->forward(x));
    x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout2));
    auto out = fc3->forward(x);
    return out;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Implementation from Scratch
	/*
	 * To implement the dropout function for a single layer, we must draw as many samples from a Bernoulli (binary) random variable as our layer has dimensions,
	 * where the random variable takes value 1 (keep) with probability 1−𝑝 and 0 (drop) with probability 𝑝. One easy way to implement this is to first draw
	 * samples from the uniform distribution 𝑈[0,1]. Then we can keep those nodes for which the corresponding sample is greater than 𝑝, dropping the rest.

	 * In the following code, we (implement a dropout_layer function that drops out the elements in the tensor input X with probability dropout), rescaling
	 * the remainder as described above: dividing the survivors by 1.0-dropout.
	 */
	auto X = torch::arange(16, dtype(torch::kDouble)).reshape({2, 8});

	std::cout << "X:\n" << X << std::endl;
	std::cout << "dropout_layer(X, 0.):\n" << dropout_layer(X, 0.) << std::endl;
	std::cout << "dropout_layer(X, 0.5):\n" << dropout_layer(X, 0.5) << std::endl;
	std::cout << "dropout_layer(X, 1.):\n" << dropout_layer(X, 1.) << std::endl;

	// Defining Model Parameters
	/*
	 * Again, we work with the Fashion-MNIST dataset introduced in :numref:sec_fashion_mnist. We [define an MLP with two hidden layers containing 256 units each.]
	 */
	int64_t num_inputs = 784, num_outputs = 10, num_hiddens1 = 256,  num_hiddens2 =256;

	// Defining the Model
	/*
	 * The model below applies dropout to the output of each hidden layer (following the activation function). We can set dropout probabilities for each layer
	 * separately. A common trend is to set a lower dropout probability closer to the input layer. Below we set it to 0.2 and 0.5 for the first and second
	 * hidden layers, respectively. We ensure that dropout is only active during training.
	 */
	//auto net = NeuralNetImpl(num_inputs, num_outputs, num_hiddens1, num_hiddens2);
	//auto net = NetCh04Impl(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true);
	//NetCh04Impl net = NetCh04Impl(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true); // using member .
	Net net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true);           // using ptr ->
	net->to(device);

	/*
	 * This is similar to the training and testing of MLPs described previously.
	 */
	int64_t num_epochs = 20;
	float lr = 0.5;
	int64_t batch_size = 256;

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
				    		.map(torch::data::transforms::Stack<>());

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
			                .map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();
	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
											         std::move(test_dataset), batch_size);

	auto criterion = torch::nn::CrossEntropyLoss();
	auto trainer = torch::optim::SGD(net->parameters(), lr);

	/*
	* Train a model
	*/

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {
		net->train(true);
		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
	    int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			//auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto x = batch.data.to(device);
			auto y = batch.target.to(device);

			//std::cout << "x.sizes(): " << x.sizes() << std::endl;

			auto y_hat = net->forward(x);
			auto loss = criterion(y_hat, y); //torch::cross_entropy_loss(y_hat, y);
				//std::cout << loss.item<double>() << std::endl;

			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);

			// Calculate prediction
			// Update number of correctly classified samples
			epoch_correct += accuracy(y_hat, y); //(prediction == y).sum().item<int>(); //prediction.eq(y).sum().item<int64_t>();
			//std::cout << epoch_correct << std::endl;
			trainer.zero_grad();
			loss.backward();
			trainer.step();

			num_train_samples += x.size(0);
		}

		auto sample_mean_loss = (epoch_loss / num_train_samples);
		auto tr_acc = static_cast<double>(epoch_correct *1.0 / num_train_samples);

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
				            << sample_mean_loss << ", Accuracy: " << tr_acc << '\n';

		train_loss.push_back((sample_mean_loss));
		train_acc.push_back(tr_acc);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for( auto& batch : *test_loader) {
			//auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			//std::cout << data.sizes() << std::endl;

			auto output = net->forward(data);

			auto loss = criterion(output, target); //torch::nn::functional::cross_entropy(output, target);
			tst_loss += loss.item<double>() * data.size(0);

			//auto prediction = output.argmax(1);
			epoch_correct += accuracy(output, target); //prediction.eq(target).sum().item<int64_t>();

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct * 1.0 / num_test_samples);
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::ylim(ax1, {0.3, 0.99});
	matplot::plot(ax1, xx, train_loss, "b")->line_width(2);
	matplot::plot(ax1, xx, test_loss, "m-:")->line_width(2);
	matplot::plot(ax1, xx, train_acc, "g--")->line_width(2);
	matplot::plot(ax1, xx, test_acc, "r-.")->line_width(2);

    matplot::hold(ax1, false);
    matplot::xlabel(ax1, "epoch");
    matplot::title(ax1, "Define an MLP with two hidden layers");
    matplot::legend(ax1, {"Train loss", "Test loss", "Train acc", "Test acc"});
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}



