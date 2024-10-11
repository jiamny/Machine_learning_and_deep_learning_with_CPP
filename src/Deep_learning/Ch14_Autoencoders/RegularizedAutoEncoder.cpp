/*
 * RegularizedAutoEncoder.cpp
 *
 *  Created on: Sep 20, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <torch/torch.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/fashion.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;
namespace F = torch::nn::functional;

#include <matplot/matplot.h>
using namespace matplot;

struct EncoderImpl : public torch::nn::Module {
	bool include_bn;
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
	torch::nn::Linear fc{nullptr};

	explicit EncoderImpl(int num_filters=128, int bottleneck_size=16, bool include_batch_norm=true) {

        include_bn = include_batch_norm;

        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, num_filters, 4)
        		.stride(2).padding({2,2}));
        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters));

        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters,num_filters * 2,4)
                .stride(2).padding({2, 2}));
        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters * 2));

        conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters * 2,num_filters * 4,4)
                .stride(2).padding({2, 2}));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters * 4));

        conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters * 4,num_filters * 8,4)
        		.stride(2).padding({2, 2}));
        bn4 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters * 8));

        fc = torch::nn::Linear(torch::nn::LinearOptions(num_filters * 72, bottleneck_size));
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("fc", fc);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("bn4", bn4);
	}

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        if(include_bn)
            x = bn1->forward(x);
        x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

        x = conv2->forward(x);
        if(include_bn)
            x = bn2->forward(x);
        x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

        x = conv3->forward(x);
        if(include_bn)
            x = bn3->forward(x);
		x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

        x = conv4->forward(x);
        if(include_bn)
            x = bn4->forward(x);
		x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

        x = x.view({x.size(0), -1});
        x = fc->forward(x);

        return x;
    }
};
TORCH_MODULE(Encoder);

struct DecoderImpl : public torch::nn::Module {
	bool include_bn;
	torch::nn::ConvTranspose2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
	torch::nn::Linear fc{nullptr};
	torch::nn::Sigmoid sigmoid{nullptr};

    explicit DecoderImpl(int num_filters=128, int bottleneck_size=16, bool include_batch_norm=true) {

        include_bn = include_batch_norm;
        conv1 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(1024,num_filters * 4, 4)
                .stride(2).padding({2, 2}));

        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters * 4));

        conv2 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(num_filters * 4, num_filters * 2, 4)
        		.stride(2).padding({2, 2}));

        bn2 =torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters * 2));

        conv3 = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(num_filters * 2, 1, 5)
        		.padding({1, 1}));

        sigmoid = torch::nn::Sigmoid();

        fc = torch::nn::Linear(torch::nn::LinearOptions(bottleneck_size, 8 * 8 * 1024));
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
        register_module("fc", fc);
        register_module("sigmoid", sigmoid);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc->forward(x);
        x = x.view({-1, 1024, 8, 8});

    	x = conv1->forward(x);
    	if(include_bn)
    	    x = bn1->forward(x);

    	x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

    	x = conv2->forward(x);
    	if(include_bn)
    	    x = bn2->forward(x);

    	x = F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.1));

    	x = conv3->forward(x);
    	x = sigmoid->forward(x);

    	return x;
    }
};
TORCH_MODULE(Decoder);

struct AutoEncoderImpl : public torch::nn::Module {
	Encoder enc{nullptr};
	Decoder dec{nullptr};

	AutoEncoderImpl(Encoder encoder, Decoder decoder) {
		enc = encoder;
		dec = decoder;
		register_module("enc", enc);
		register_module("dec", dec);
	}

	torch::Tensor forward(torch::Tensor x) {
		int batch_size = x.size(0);
		// encoder
		x = enc->forward(x);
		// decoder
		x = dec->forward(x);

		// reshape
		x = x.view({batch_size, 1, 28, 28});

		return x;
	}
};

TORCH_MODULE(AutoEncoder);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device = torch::Device(torch::kCPU);

	if( cuda_available ) {
		int gpu_id = 0;
		device = torch::Device(torch::kCUDA, gpu_id);

		if(gpu_id >= 0) {
			if(gpu_id >= torch::getNumGPUs()) {
				std::cout << "No GPU id " << gpu_id << " abailable, use CPU." << std::endl;
				device = torch::Device(torch::kCPU);
				cuda_available = false;
			} else {
				device = torch::Device(torch::kCUDA, gpu_id);
			}
		} else {
			device = torch::Device(torch::kCPU);
			cuda_available = false;
		}
	}

	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
	std::cout << device << '\n';

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

	// Create the model
    auto encoder = Encoder();
    auto decoder = Decoder();
    encoder->to(device);
    decoder->to(device);

	torch::Tensor a = torch::randn({1, 1, 28, 28}).to(device);
	std::cout << a.sizes() << '\n';

	AutoEncoder model = AutoEncoder(encoder, decoder);
	std::cout << model->forward(a).sizes() << '\n';

	std::tuple<double,double>  betas = {0.5, 0.999};
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(5e-4)
    		.betas(betas).weight_decay(0.0001));

    auto criteria = torch::nn::MSELoss();

    int EPOCHES = 10;
     model->train();
     for(auto& epoch : range(EPOCHES, 0) ) {
     	float ls = 0.;
 		for (auto& batch : *train_loader) {
 			auto img = batch.data.to(device);
 			auto _ = batch.target.to(device);
 			auto out = model->forward(img.clone());
 			auto loss = criteria(out, img);
 			optimizer.zero_grad();
 			loss.backward();
 			optimizer.step();
 			ls += loss.data().item<float>();
 		}
 		printf("Epoch: %2d/%2d, loss: %.4f\n", (epoch+1), EPOCHES, ls);
     }

     // 将生成图片和原始图片进行对比
     model->eval();
     int i = 0;
 	for (auto& batch : *test_loader) {
 		auto img = batch.data;
 		auto label = batch.target;
     	torch::Tensor img_new = model->forward(img.to(device)).detach().cpu();

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



