/*
 * FashionImgGenerateWithGAN.cpp
 *
 *  Created on: Jul 30, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/opencv_helpfunctions.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/fashion.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

class Discriminator : public torch::nn::Module {
public:
	Discriminator() {
        model = torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(784, 1024)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Dropout(torch::nn::DropoutOptions(0.3)),
			torch::nn::Linear(torch::nn::LinearOptions(1024, 512)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Dropout(torch::nn::DropoutOptions(0.3)),
			torch::nn::Linear(torch::nn::LinearOptions(512, 256)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Dropout(torch::nn::DropoutOptions(0.3)),
			torch::nn::Linear(torch::nn::LinearOptions(256, 1)),
            torch::nn::Sigmoid()
        );
        register_module("model", model);
	}

	torch::Tensor forward(torch::Tensor x) {
		return model->forward(x);
	}

private:
	torch::nn::Sequential model{nullptr};
};

class Generator : public torch::nn::Module {
public:
	Generator() {
        model = torch::nn::Sequential(
        	torch::nn::Linear(torch::nn::LinearOptions(100, 256)),
        	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
			torch::nn::Linear(torch::nn::LinearOptions(256, 512)),
			torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
			torch::nn::Linear(torch::nn::LinearOptions(512, 1024)),
			torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
			torch::nn::Linear(torch::nn::LinearOptions(1024, 784)),
            torch::nn::Tanh()
        );
        register_module("model", model);
	}

	torch::Tensor forward(torch::Tensor x) {
		return model->forward(x);
	}

private:
	torch::nn::Sequential model{nullptr};
};

torch::Tensor noise(int size, torch::Device device) {
	torch::Tensor n = torch::randn({size, 100});
    return n.to(device);
}

torch::Tensor discriminator_train_step(Discriminator& discriminator, torch::optim::Adam& d_optimizer,
		torch::nn::BCELoss& loss, torch::Tensor real_data, torch::Tensor fake_data) {
    d_optimizer.zero_grad();
    torch::Tensor prediction_real = discriminator.forward(real_data);
	torch::Tensor error_real = loss(prediction_real,
			torch::ones({real_data.size(0), 1}).to(real_data.device()));

    error_real.backward();
    torch::Tensor prediction_fake = discriminator.forward(fake_data);
    torch::Tensor error_fake = loss(prediction_fake,
    		torch::zeros({fake_data.size(0), 1}).to(fake_data.device()));
    error_fake.backward();
    d_optimizer.step();
    return error_real + error_fake;
}

torch::Tensor generator_train_step(Discriminator& discriminator, torch::optim::Adam& g_optimizer,
									torch::nn::BCELoss& loss, torch::Tensor fake_data) {
    g_optimizer.zero_grad();
	torch::Tensor  prediction = discriminator.forward(fake_data);
	torch::Tensor error = loss(prediction, torch::ones({fake_data.size(0), 1}).to(fake_data.device()));
    error.backward();
    g_optimizer.step();
    return error;
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

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Load data\n";
	std::cout << "// --------------------------------------------------\n";

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

	int num_epochs = 200; // train the training data n times, to save time, we just train 1 epoch
	int batch_size = 128;

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	FASHION train_datas(FASHION_data_path, FASHION::Mode::kTrain);
	auto train_dataset = train_datas.map(torch::data::transforms::Stack<>()); //.map(torch::data::transforms::Normalize(mean, stddev));

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Train and generate image\n";
	std::cout << "// --------------------------------------------------\n";

	Discriminator discriminator = Discriminator();
	discriminator.to(device);
	Generator generator = Generator();
	generator.to(device);

	torch::optim::Adam d_optimizer = torch::optim::Adam(discriminator.parameters(), 0.0002);
	torch::optim::Adam g_optimizer = torch::optim::Adam(generator.parameters(), 0.0002);
	torch::nn::BCELoss loss = torch::nn::BCELoss();

	float mean = 0.5, stddev = 0.5;
	std::vector<double> d_losses, g_losses, x_epochs;

	for(auto& epoch : range(num_epochs, 0) ) {
	    int N = 0;
	    double d_L = 0., g_L = 0.;
	    for (auto& batch : *train_loader) {
	    	auto real_data =  batch.data.squeeze().to(device);
	    	real_data = real_data.view({real_data.size(0), -1}).to(device);
	    	// ---------------------------------------------------------
	    	// Normalizes input tensors by subtracting the supplied mean
	    	// and dividing by the given standard deviation.
	    	// ----------------------------------------------------------
	    	real_data = real_data.sub(mean).div(stddev);

	    	//std::cout << real_data.sizes() << '\n';
	    	torch::Tensor fake_data = generator.forward(noise(real_data.size(0), device)).to(device);
	    	//std::cout << fake_data.sizes() << '\n';
	    	fake_data = fake_data.detach();

	    	torch::Tensor d_loss = discriminator_train_step(discriminator, d_optimizer,
	    													loss, real_data, fake_data);

	    	fake_data = generator.forward(noise(real_data.size(0), device)).to(device);
	    	torch::Tensor g_loss = generator_train_step(discriminator, g_optimizer, loss, fake_data);
	    	d_L += d_loss.data().item<float>();
	    	g_L += g_loss.data().item<float>();
	    	N++;
	    }

	    d_losses.push_back(d_L/N);
	    g_losses.push_back(g_L/N);
	    x_epochs.push_back((epoch+1)*1.0);
	    printf("Epoch %3d/%d\tAvg. discriminator loss:%04f\tAvg. generator loss:%04f\n",
	    		(epoch+1), num_epochs, d_L/N, g_L/N);
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	matplot::plot(ax, x_epochs, d_losses, "b-")->line_width(2).display_name("Discriminator loss");
	matplot::plot(ax, x_epochs, g_losses, "r--")->line_width(2).display_name("Generator loss");
    matplot::hold(ax, false);
    matplot::xlabel(ax, "epoch");
    matplot::legend(ax, {});
    matplot::show();

    torch::Tensor z = torch::randn({24, 100}).to(device);
    torch::Tensor sample_images = generator.forward(z).data().cpu().view({24 , 1, 28, 28});

	F = figure(true);
	F->size(1000, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);

	for(auto& j : range(24, 0)) {
		std::vector<std::vector<double>>  oimg = get_mnist_image(sample_images[j].squeeze());
		matplot::subplot(3, 8, j);
		matplot::axis(false);
		matplot::image(oimg);
	}
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




