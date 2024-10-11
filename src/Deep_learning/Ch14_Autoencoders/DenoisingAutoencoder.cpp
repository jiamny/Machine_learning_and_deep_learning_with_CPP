/*
 * DenoisingAutoencoder.cpp
 *
 *  Created on: Sep 22, 2024
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


torch::Tensor add_noise(torch::Tensor img, std::string noise_type="gaussian") {

	//img = img.to(torch::kDouble);
    int row = 28, col = 28;
    if( noise_type=="gaussian" ) {
    	double mean = 0.;
	    double var = 10.0;
	    double sigma = std::sqrt(var);
	    torch::Tensor noise = torch::normal(mean, sigma, img.sizes()).to(img.dtype());
	    noise = noise.reshape({row, col}).div(255);
	    img = img + img*noise;
    }

    if( noise_type=="speckle" ) {
    	torch::Tensor noise = torch::randn(img.sizes()).to(img.dtype());
        noise = noise.reshape({row, col}).div(255);
        img = img + img*noise;
    }

    return img;
}

void show_images(torch::Tensor orgImg1, torch::Tensor orgImg2, torch::Tensor noisedImg1,
				torch::Tensor noisedImg2, std::string label1, std::string label2) {

	std::vector<std::vector<double>> C = get_mnist_image( orgImg1.squeeze() );
	auto F1 = figure(true);
	F1->size(1000, 800);
	F1->add_axes(false);
	F1->reactive_mode(false);
	F1->tiledlayout(2, 2);
	F1->position(0, 0);

	auto ax1 = F1->nexttile();
	matplot::image(ax1, C);
	matplot::title(ax1, "Original Image: " + label1);

	std::vector<std::vector<double>> C2 = get_mnist_image( orgImg2.squeeze() );
	auto ax2 = F1->nexttile();
	matplot::image(ax2, C2);
	matplot::title(ax2, "Original Image: " + label2);

	std::vector<std::vector<double>> C3 = get_mnist_image( noisedImg1.squeeze() );
	auto ax3 = F1->nexttile();
	matplot::image(ax3, C3);
	matplot::title(ax3, "Noised Image: " + label1);

	std::vector<std::vector<double>> C4 = get_mnist_image( noisedImg2.squeeze() );
	auto ax4 = F1->nexttile();
	matplot::image(ax4, C4);
	matplot::title(ax4, "Noised Image: " + label2);

	matplot::show();
}

struct DenoisingAuutoEncoderImpl : public torch::nn::Module {
	torch::nn::Sequential encoder{nullptr}, decoder{nullptr};
	DenoisingAuutoEncoderImpl() {
		encoder = torch::nn::Sequential(
				torch::nn::Linear(torch::nn::LinearOptions(28*28,256)),
				torch::nn::ReLU(torch::nn::ReLUOptions(true)),
				torch::nn::Linear(torch::nn::LinearOptions(256,128)),
				torch::nn::ReLU(torch::nn::ReLUOptions(true)),
				torch::nn::Linear(torch::nn::LinearOptions(128,64)),
				torch::nn::ReLU(torch::nn::ReLUOptions(true)));

    	decoder = torch::nn::Sequential(
    			torch::nn::Linear(torch::nn::LinearOptions(64,128)),
				torch::nn::ReLU(torch::nn::ReLUOptions(true)),
				torch::nn::Linear(torch::nn::LinearOptions(128,256)),
				torch::nn::ReLU(torch::nn::ReLUOptions(true)),
				torch::nn::Linear(torch::nn::LinearOptions(256,28*28)),
				torch::nn::Sigmoid());

		register_module("enc", encoder);
		register_module("dec", decoder);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = encoder->forward(x);
    	x = decoder->forward(x);

		return x;
	}
};
TORCH_MODULE(DenoisingAuutoEncoder);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(345);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device = torch::Device(torch::kCPU);
	std::cout << "cuda_available " << cuda_available << '\n';

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

	// Number of samples in the training set
	int num_train_samples = train_datas.images().size(0);
	std::cout << "num_train_samples: " << num_train_samples << std::endl;
	torch::Tensor xtrain = train_datas.images();
	torch::Tensor ytrain = train_datas.targets();

	std::vector<std::string> noises = {"gaussian", "speckle"};
	int noise_ct = 0;
	int noise_id = 0;
	torch::Tensor traindata = torch::zeros(xtrain.sizes()).to(xtrain.dtype());

	for(auto& idx : range(num_train_samples, 0)) {

		if( noise_ct < static_cast<int>((num_train_samples / 2.)) ) {
			noise_ct += 1;
			traindata.index_put_({idx, 0, Slice(), Slice()}, add_noise(xtrain[idx].squeeze(), noises[noise_id]));

		} else {
			printf("%s noise addition completed to images\n", noises[noise_id].c_str());
			noise_id += 1;
			noise_ct = 0;
		}
	}
	printf("%s noise addition completed to images\n", noises[noise_id].c_str());

	FASHION test_datas(FASHION_data_path, FASHION::Mode::kTest);

	int num_test_samples = test_datas.images().size(0);
	std::cout << "num_test_samples: " << num_test_samples << std::endl;
	torch::Tensor xtest = test_datas.images();
	torch::Tensor ytest = test_datas.targets();

	noise_ct = 0;
	noise_id = 0;
	torch::Tensor testdata = torch::zeros(xtest.sizes()).to(xtest.dtype());

	for(auto& idx : range(num_test_samples, 0)) {

		if( noise_ct < static_cast<int>((num_test_samples / 2)) ) {
			noise_ct += 1;
	    	testdata.index_put_({idx, 0, Slice(), Slice()}, add_noise(xtest[idx].squeeze(), noises[noise_id]));

		} else {
			printf("%s noise addition completed to images\n", noises[noise_id].c_str());
			noise_id += 1;
			noise_ct = 0;
		}
	}
	printf("%s noise addition completed to images\n", noises[noise_id].c_str());


	torch::Tensor img1 = xtrain[0].clone();
	int label_id = ytrain[0].data().item<int>();
	torch::Tensor n_img1 = traindata[0].clone();
	std::string label1 = fashionMap[label_id];

	torch::Tensor img2 = xtrain[31000].clone();
	torch::Tensor n_img2 = traindata[31000].clone();
	label_id = ytrain[31000].data().item<int>();
	std::string label2 = fashionMap[label_id];
	show_images(img1.mul(255), img2.mul(255), n_img1, n_img2, label1, label2);

	DenoisingAuutoEncoder model = DenoisingAuutoEncoder();

	model->to(device);
	auto criterion = torch::nn::MSELoss();
	auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.01).weight_decay(1e-5));

	int epochs = 200;
	int64_t batch_size = 32;

	std::vector<double> losslist, epochx;
	double running_loss = 0;

	for(auto& epoch : range(epochs, 0) ) {
		std::list<torch::Tensor> dataIdxs = data_index_iter(static_cast<int64_t>(xtrain.size(0)), batch_size);
		int l = dataIdxs.size();

		for(auto& batch_idx : dataIdxs) {
			torch::Tensor dirty = traindata.index_select(0, batch_idx).clone().view({batch_idx.size(0),-1});
			torch::Tensor clean = xtrain.index_select(0, batch_idx).clone().view({batch_idx.size(0), -1});

			dirty = dirty.to(device);
			clean = clean.to(device);

			// -----------------Forward Pass----------------------
			torch::Tensor output = model->forward(dirty);
		    auto loss = criterion(output, clean);
			// -----------------Backward Pass---------------------
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

		    running_loss += loss.data().item<float>();
		}
		// -----------------Log-------------------------------
		losslist.push_back(running_loss/l);
		running_loss = 0;
		printf("======> epoch: %3d / %3d, Loss:%2.4f\n", epoch, epochs, losslist[epoch]);
		epochx.push_back(epoch * 1.0);
	}

 	auto F = figure(true);
 	F->size(800, 600);
 	F->add_axes(false);
 	F->reactive_mode(false);
 	F->position(0, 0);
 	auto ax = F->nexttile();
 	matplot::plot(ax, epochx, losslist, "m-")->line_width(2);
 	matplot::xlabel("epoch");
 	matplot::xlabel("loss");
 	matplot::show();

	auto h = figure(true);
	h->size(1500, 600);
	h->position(0, 0);

	torch::Tensor test_imgs = torch::randint(0,10000,{5});

	for(auto & i : range(5, 0)) {
		matplot::subplot(3, 5, i);
		int yid = test_imgs[i].data().item<int>();
		int type_id = ytest[yid].data().item<int>();
		std::string label = fashionMap[type_id];

		torch::Tensor dirty = testdata[yid].clone();
		torch::Tensor clean = xtest[yid].clone();

		matplot::title("Original Img: " + label);
		std::vector<std::vector<double>> C = get_mnist_image( clean.squeeze() );
		matplot::image(C);

		matplot::subplot(3, 5, i + 5);
		matplot::title("Dirty Img: " + label);
		std::vector<std::vector<double>> C2 = get_mnist_image( dirty.squeeze() );
		matplot::image(C2);

		dirty = dirty.to(device);
		clean = clean.to(device);

		torch::Tensor output = model->forward(dirty.view({1,-1})).cpu();
		output = output.reshape({28, 28});
		matplot::subplot(3, 5, i + 10);
		matplot::title("Cleaned Img: " + label);
		std::vector<std::vector<double>> C3 = get_mnist_image( output.squeeze() );
		matplot::image(C3);
	}
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}
