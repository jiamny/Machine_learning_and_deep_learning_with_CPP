/*
 * AutoEncoder.cpp
 *
 *  Created on: Jul 18, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/fashion.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace F = torch::nn::functional;

#include <matplot/matplot.h>
using namespace matplot;

class AutoEncoder : public torch::nn::Module {
public:
	 AutoEncoder() {
        // 2层卷积神经网络编码器
        encoder = torch::nn::Sequential();
        encoder->push_back("conv2d_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)));
        encoder->push_back("enrelu_1", torch::nn::ReLU());
        encoder->push_back("conv2d_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
        encoder->push_back("enrelu_2", torch::nn::ReLU());
        // 2层卷积神经网络解码器
        decoder = torch::nn::Sequential();
        decoder->push_back("convts_1",
        		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 16, 3).stride(2).padding(1).output_padding(1)));
        decoder->push_back("derelu_1", torch::nn::ReLU());
		decoder->push_back("convts_2",
				torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(16, 1, 3).stride(2).padding(1).output_padding(1)));
		decoder->push_back("derelu_2", torch::nn::Sigmoid());
        register_module("encoder", encoder);
        register_module("decoder", decoder);
	 }

	 torch::Tensor forward(torch::Tensor x) {
        x = encoder->forward(x);
        x = decoder->forward(x);
        return x;
    }
private:
	 torch::nn::Sequential encoder{nullptr}, decoder{nullptr};
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

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

	float learning_rate = 0.01;

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	FASHION train_datas(FASHION_data_path, FASHION::Mode::kTrain);
	auto train_dataset = train_datas.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), 32);

	FASHION test_datas(FASHION_data_path, FASHION::Mode::kTest);
	auto test_dataset = test_datas.map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(test_dataset), 8);

	AutoEncoder model = AutoEncoder();
	model.to(device);
    std::cout << model << '\n';

    // 设置损失函数
    auto criterion = torch::nn::MSELoss();
    // 设置优化器
    auto optimizer = torch::optim::Adam(model.parameters(), learning_rate);

    // 模型训练
    int EPOCHES = 20;
    model.train();
    for(auto& epoch : range(EPOCHES, 0) ) {
    	float ls = 0.;
		for (auto& batch : *train_loader) {
			auto img = batch.data.to(device);
			auto _ = batch.target.to(device);
			auto out = model.forward(img.clone());
			auto loss = criterion(out, img);
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			ls += loss.data().item<float>();
		}
		printf("Epoch: %2d/%2d, loss: %.4f\n", (epoch+1), EPOCHES, ls);
    }

    // 将生成图片和原始图片进行对比
    model.eval();
    int i = 0;
	for (auto& batch : *test_loader) {
		auto img = batch.data;
		auto label = batch.target;
    	torch::Tensor img_new = model.forward(img.to(device)).detach().cpu();

    	auto F = figure(true);
    	F->size(1000, 400);
    	F->add_axes(false);
    	F->reactive_mode(false);
    	F->position(0, 0);

    	for(auto& j : range(8, 0)) {
    		int type_id = label[j].data().item<int>();
    		std::string it = fashionMap[type_id];
    		std::vector<std::vector<double>>  oimg = get_mnist_image(img[j].squeeze());
    		std::vector<std::vector<double>>  nimg = get_mnist_image(img_new[j].squeeze());
    		matplot::subplot(2, 8, j);
    		matplot::axis(false);
    		matplot::image(oimg);
    		matplot::title(it.c_str());
    		matplot::subplot(2, 8, 8 + j);
    		matplot::axis(false);
    		matplot::image(nimg);
    	}
		if(i >= 2)
			break;
		i++;
		matplot::show();
	}

	std::cout << "Done!\n";
	return 0;
}
