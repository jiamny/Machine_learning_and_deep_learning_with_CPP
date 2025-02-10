/*
 * C10_5_Convolution_for_MNIST.cpp
 *
 *  Created on: Jan 31, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/opencv_helpfunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// Deep Convolutional Neural Networks
/*
 # 1. A valid convolution with kernel size 5, 1 input channel and 10 output channels
 # 2. A max pooling operation over a 2x2 area
 # 3. A Relu
 # 4. A valid convolution with kernel size 5, 10 input channels and 20 output channels
 # 5. A 2D Dropout layer
 # 6. A max pooling operation over a 2x2 area
 # 7. A relu
 # 8. A flattening operation
 # 9. A fully connected layer mapping from (whatever dimensions we are at-- find out using .shape) to 50
 # 10. A ReLU
 # 11. A fully connected layer mapping from 50 to 10 dimensions
 # 12. A softmax function.
 */

struct Net : torch::nn::Module {
  Net() {
		model = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
				torch::nn::ReLU(),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)),
				torch::nn::Dropout(0.5),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
				torch::nn::ReLU(),
				torch::nn::Flatten(),
				torch::nn::Linear(torch::nn::LinearOptions(320, 50)),
				torch::nn::ReLU(),
				torch::nn::Linear(torch::nn::LinearOptions(50, 10)));
		register_module("model", model);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = model->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Sequential model;
};

template <typename Net, typename DataLoader>
void train(
    size_t epoch,
	Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % 5 == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.5f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename Net, typename DataLoader>
void test(
	Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.5f | Accuracy: %.5f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Loading data\n";
	std::cout << "// ------------------------------------------------------------------------\n";

    const char* kDataRoot = "/media/hhj/localssd/DL_data/mnist2/MNIST/raw";

    // The batch size for training.
    const int64_t kTrainBatchSize = 64;
    // The batch size for testing.
    const int64_t kTestBatchSize = 1000;
    // The number of epochs to train.
    const int64_t kNumberOfEpochs = 10;

    Net model;
    model.to(device);

    auto X = torch::randn({256, 1, 28, 28}).to(torch::kFloat32);
    	std::cout << model.forward(X.to(device)).sizes() << '\n';

    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), kTrainBatchSize);

    auto test_dataset = torch::data::datasets::MNIST(
                            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
        torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
      test(model, device, *test_loader, test_dataset_size);
    }

    torch::NoGradGuard no_grad;
    model.eval();

    auto batch = *test_loader->begin();
    torch::Tensor data = batch.data.to(device);
    torch::Tensor targets = batch.target;
    std::cout << data.sizes() << '\n';
    torch::Tensor output = model.forward(data).detach().cpu();
    std::optional<long int> ax = {1};
    torch::Tensor pred = output.argmax(ax);

	auto F = figure(true);
	F->size(1000, 400);
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);
	data = data.detach().cpu();

	for(auto& j : range(16, 0)) {
		int type_id = pred[j].data().item<int>();
		std::string it = "Pred: " + std::to_string(type_id);
		torch::Tensor dnm_img =  deNormalizeTensor(data[j], {0.1307}, {0.3081});
		std::vector<std::vector<double>>  oimg = get_mnist_image(dnm_img.squeeze());
		matplot::subplot(2, 8, j);
		matplot::image(oimg);
		matplot::axis(false);
		matplot::title(it.c_str());
	}
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





