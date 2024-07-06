/*
 * ConvolutionalNetworks.cpp
 *
 *  Created on: Jun 14, 2024
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

#include "../../Utils/TempHelpFunctions.h"
#include "../../Algorithms/PrincipalComponentsAnalysis.h"
#include "../../Utils/fashion.h"
#include "../../Utils/helpfunction.h"


#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

class CNN : public torch::nn::Module {
public:
	torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
	torch::nn::Linear out{nullptr};

    CNN(void) {
        conv1 = torch::nn::Sequential(         // input shape (1, 28, 28)
        		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)
                //1,              // input height
                //16,             // n_filters, out_channels
                //5,              // filter size, kernel_size
                //1,              // filter movement/step, stride
                //2,              // padding, if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              // output shape (16, 28, 28)
            torch::nn::ReLU(),                      // activation
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))    // kernel_size, choose max value in 2x2 area, output shape (16, 14, 14)
        );

        conv2 = torch::nn::Sequential(         // input shape (1, 14, 14)
        	torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),     // output shape (32, 14, 14)
			torch::nn::ReLU(),                      // activation
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))             // output shape (32, 7, 7)
        );
        out = torch::nn::Linear(torch::nn::LinearOptions(32 * 7 * 7, 10));   // fully connected layer, output 10 classes
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("out", out);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = conv2->forward(x);
        x = x.view({x.size(0), -1});           // flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        torch::Tensor output = out->forward(x);
        return std::make_pair(output, x);   	// return x for visualization
    }
};


void drawFigure(torch::Tensor X_s, torch::Tensor y_s, torch::Tensor T_s, int epoch, axes_handle ax2) {
	torch::Tensor unI = std::get<0>(torch::_unique(y_s));
	//std::cout << "unI: " << unI.size(0);
	//printVector(tensorTovector(unI.to(torch::kDouble)));

	colormap(palette::paired(unI.size(0)));
	auto cmap = colormap();

	torch::Tensor midx = RangeTensorIndex(y_s.size(0));

	for(auto& i : range(static_cast<int>(unI.size(0)), 0)) {
		torch::Tensor msk = (y_s == unI[i]);
		//std::cout << "msk: " << msk.sizes() << '\n';
		auto mid_x = midx.masked_select(msk);
			//std::cout << "id_x: " << id_x.sizes() << '\n';
		//printVector(tensorTovector(mid_x.to(torch::kDouble)));
		auto mtst_x = torch::index_select(X_s, 0, mid_x.squeeze());
		//std::cout << "mtst_x: " << mtst_x.sizes() << '\n';
		//auto mtst_y = test_y.masked_select(msk);
			//std::cout << "tst_y[0]: " << tst_y[0] << '\n';
		auto mt_s = torch::index_select(T_s, 0, mid_x.squeeze());

		std::vector<double> xx = tensorTovector(mt_s.index({Slice(), 0}).to(torch::kDouble));
		std::vector<double> yy = tensorTovector(mt_s.index({Slice(), 1}).to(torch::kDouble));
		for(auto& j : range(static_cast<int>(xx.size()), 0)) {
			matplot::text(ax2, xx[j], yy[j],
					std::to_string(unI[i].data().item<int>()))->color(cmap[i]).font_size(15);
		}
	}
	title(ax2, "After epoch: " + std::to_string(epoch) + " training");
	ax2->draw();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';


    // Create an unordered_map to hold label names
    std::map<int, std::string> fashionMap = {
    		{0, "T-shirt/top"},
			{1, "Trouser"},
			{2, "Pullover"},
			{3, "Dress"},
			{4, "Coat"},
			{5, "Sandal"},
			{6, "Short"},
			{7, "Sneaker"},
			{8, "Bag"},
			{9, "Ankle boot"}};

    bool show_image = true;

	int num_epochs = 20; // train the training data n times, to save time, we just train 1 epoch
	int batch_size = 50;

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	FASHION train_datas(FASHION_data_path, FASHION::Mode::kTrain);

	if (show_image ) {
		at::Tensor im = train_datas.images()[11].squeeze();
		torch::Tensor target = train_datas.targets()[11].squeeze();

		int type_id = target.data().item<int>();
		std::cout << "type_id = " << type_id << " name = " << fashionMap.at(type_id) << "\n";

		std::vector<std::vector<double>>  C = get_mnist_image(im);
		auto F = figure(true);
		F->size(600, 600);
		F->add_axes(false);
		F->reactive_mode(false);
		F->tiledlayout(1, 1);
		F->position(0, 0);


		auto ax = F->nexttile();
		matplot::image(ax, C);
		auto search = fashionMap.find(type_id);
		if (search != fashionMap.end()) {
			std::string it = fashionMap[type_id];
			matplot::title(ax, it.c_str());
		}
		matplot::show();
	}

	auto train_dataset = train_datas.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);


	FASHION test_data(FASHION_data_path, FASHION::Mode::kTest);
	at::Tensor test_x = train_datas.images().index({Slice(0, 2000), Slice()});
	torch::Tensor test_y = train_datas.targets().index({Slice(0, 2000)});

	if (show_image ) {
		torch::Tensor mask = (test_y == 4);
		std::cout << "mask: " << mask.sizes() << '\n';
		torch::Tensor idx = RangeTensorIndex(test_y.size(0));
		auto id_x = idx.masked_select(mask);
		std::cout << "id_x: " << id_x.sizes() << '\n';
		printVector(tensorTovector(id_x.to(torch::kDouble)));
		auto tst_x = torch::index_select(test_x, 0, id_x.squeeze());
		std::cout << "tst_x: " << tst_x.sizes() << '\n';
		auto tst_y = test_y.masked_select(mask);
		std::cout << "tst_y[0]: " << tst_y[0] << '\n';

		std::vector<std::vector<double>> C = get_mnist_image(tst_x[0].squeeze());
		auto F1 = figure(true);
		F1->size(600, 600);
		F1->add_axes(false);
		F1->reactive_mode(false);
		F1->tiledlayout(1, 1);
		F1->position(0, 0);

		auto ax1 = F1->nexttile();
		matplot::image(ax1, C);
		matplot::title(ax1, fashionMap.at(4).c_str());
		matplot::show();
	}

	float LR = 0.001;				// learning rate

	CNN cnn = CNN();
	std::cout << cnn << '\n';		// net architecture
	cnn.to(device);

	auto optimizer = torch::optim::Adam(cnn.parameters(), LR);   // optimize all cnn parameters
	auto loss_func = torch::nn::CrossEntropyLoss();              // the target label is not one-hotted

	auto F2 = figure(true);
	F2->size(1200, 1000);
	F2->add_axes(false);
	F2->reactive_mode(false);
	F2->tiledlayout(1, 1);
	F2->position(0, 0);
	auto ax2 = F2->nexttile();

	for(auto& epoch : range(num_epochs, 0)) {

		cnn.train();
		float ls = 0.0;
		int step = 0;
		for (auto& batch : *train_loader) {

		    // Transfer images and target labels to device
			auto b_x = batch.data.to(device);
		    auto b_y = batch.target.to(device);

	        auto output = cnn.forward(b_x).first;	// cnn output
	        auto loss = loss_func(output, b_y);		// cross entropy loss
	        optimizer.zero_grad();           		// clear gradients for this training step
	        loss.backward();						// backpropagation, compute gradients
	        optimizer.step();						// apply gradients
	        ls += loss.data().item<float>();
	        step += 1;
		}

		cnn.eval();
    	torch::Tensor test_output, last_layer;
        std::tie(test_output, last_layer) = cnn.forward(test_x.to(device));
        std::optional<long int>dim = {1};
		torch::Tensor pred_y = torch::argmax(test_output, dim);
        torch::Tensor accuracy = torch::sum(pred_y == test_y.to(device)) *1.0 / test_y.size(0);
        printf("Epoch: %d | train loss: %.4f, | test accuracy: %.2f\n", (epoch+1),
        		(ls/step), accuracy.data().item<float>());

		PCA pca(2);
		torch::Tensor X_s = last_layer.cpu().index({Slice(0, 200), Slice()});
		torch::Tensor y_s = test_y.index({Slice(0, 200)});
		torch::Tensor T_s = pca.fit_transform(X_s);
		std::cout << "// --------------------------------------------------\n";
		std::cout << "// reduction last_layer data with PCA\n";
		std::cout << "// --------------------------------------------------\n";

		drawFigure(X_s, y_s, T_s, (epoch+1), ax2);
	}
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





