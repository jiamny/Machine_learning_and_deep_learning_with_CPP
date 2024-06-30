/*
 * CNN_classifying.cpp
 *
 *  Created on: Jun 21, 2024
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

torch::Tensor conv1d(torch::Tensor x, torch::Tensor w, int p=0, int s=1) {
	torch::Tensor w_rot = torch::flip(w, 0); //np.array(w[::-1])
	torch::Tensor x_padded = x.clone();

	if(p > 0) {
		torch::Tensor zero_pad = torch::zeros({p});
		x_padded = torch::cat({zero_pad, x_padded, zero_pad}, 0);
	}

	std::vector<torch::Tensor> res;
	for(int i = 0;  i < static_cast<int>((x_padded.size(0) - w_rot.size(0))) + 1; i += s) {
		res.push_back(torch::sum(x_padded.index({Slice(i, i+w_rot.size(0))}) * w_rot).unsqueeze(0));
	}

    return torch::cat(res, 0);
}

torch::Tensor conv2d(torch::Tensor X, torch::Tensor W, std::vector<int> P={0, 0}, std::vector<int> S={1, 1}) {

	torch::Tensor W_rot = torch::flip(W, {0, 1});
	torch::Tensor X_orig = X.clone();

    int n1 = X_orig.size(0) + 2*P[0];
    int n2 = X_orig.size(1) + 2*P[1];
	torch::Tensor  X_padded = torch::zeros({n1, n2}, torch::kFloat32);
    X_padded.index_put_({Slice(P[0], P[0]+X_orig.size(0)), Slice(P[1], P[1]+X_orig.size(1))}, X_orig);

    std::vector<float> res;
    int r = 0, c = 0;
    for(int i = 0; i < (static_cast<int>((X_padded.size(0) - W_rot.size(0))*1.0/S[0])+1); i += S[0]) {
    	c = 0;
        for(int j = 0; j < (static_cast<int>((X_padded.size(1) - W_rot.size(1))*1.0/S[1])+1); j +=S[1]) {
        	torch::Tensor X_sub = X_padded.index({Slice(i, i+W_rot.size(0)), Slice(j, j+W_rot.size(1))});
            res.push_back(torch::sum(X_sub * W_rot).data().item<float>());
            c++;
        }
        r++;
    }

    return torch::from_blob(res.data(), {r, c}, c10::TensorOptions(torch::kFloat32)).clone();
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

	// Testing:
	torch::Tensor x = torch::tensor({1., 3., 2., 4., 5., 6., 1., 3.});
	torch::Tensor w = torch::tensor({1., 0., 3., 1., 2.});
	int p = 2, s = 1;

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Conv1d Implementation:\n";
	std::cout << "// -----------------------------------------------------------------\n";
    torch::Tensor t = conv1d(x, w, p, s);
    printVector(tensorTovector(t.to(torch::kDouble)));

	torch::Tensor X = torch::tensor({{1., 3., 2., 4.}, {5., 6., 1., 3.}, {1., 2., 0., 2.}, {3., 4., 3., 2.}});
	torch::Tensor W = torch::tensor({{1., 0., 3.}, {1., 2., 1.}, {0., 1., 1.}});

	std::vector<int> P = {1, 1};
	std::vector<int> S = {1, 1};
	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Conv2d Implementation:\n";
	std::cout << "// -----------------------------------------------------------------\n";
	torch::Tensor t2 = conv2d(X, W, P, S);
	printVector(tensorTovector(t2.to(torch::kDouble)));

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Neural network with L2 regularization\n";
	std::cout << "// -----------------------------------------------------------------\n";
    std::string path = "./data/example-image.png";
    cv::Mat mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

    cv::imshow("origin BGR image", mat);
    cv::waitKey(0);
    cv::destroyAllWindows();

    torch::Tensor img = load_image(path.c_str());

    std::cout << "Image shape: " << img.sizes() << '\n';
    std::cout << "Number of channels: " << img.size(0) << " Number of rows/H: " << img.size(1)
    		  << " Number of cols/W: "  << img.size(2) << '\n';
    std::cout << "Image data type: " << img.dtype() << '\n';
    std::cout << "img[:, 100:102, 100:102]" << img.index({Slice(), Slice(100, 102), Slice(100, 102)}) << '\n';

    auto loss_func = torch::nn::BCELoss();

    auto loss = loss_func(torch::tensor({0.9}), torch::tensor({1.0}));
    float l2_lambda = 0.001;

    torch::nn::Conv2d conv_layer = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 5, 5)); //in_channels=3, out_channels=5, kernel_size=5)
    float l2_penalty = 0.;
    for(auto& pr : conv_layer->parameters(false)) {
    	//std::cout << p << '\n';
    	l2_penalty += torch::pow(pr, 2).sum().data().item<float>();
    }
    l2_penalty = l2_penalty * l2_lambda;
    float loss_with_penalty = loss.data().item<float>() + l2_penalty;
    std::cout << "loss_with_penalty: " << loss_with_penalty << '\n';


    torch::nn::Linear linear_layer = torch::nn::Linear(torch::nn::LinearOptions(10, 16));
    l2_penalty = 0.;
    for(auto& pr : linear_layer->parameters(false)) {
    	//std::cout << p << '\n';
    	l2_penalty += torch::pow(pr, 2).sum().data().item<float>();
    }
    l2_penalty = l2_penalty * l2_lambda;
    loss_with_penalty = loss.data().item<float>() + l2_penalty;
    std::cout << "loss_with_penalty: " << loss_with_penalty << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Binary Cross-entropy\n";
	std::cout << "// -----------------------------------------------------------------\n";
	torch::Tensor logits = torch::tensor({0.8});
	torch::Tensor probas = torch::sigmoid(logits);
	torch::Tensor target = torch::tensor({1.0});

	auto bce_loss_fn = torch::nn::BCELoss();
	auto bce_logits_loss_fn = torch::nn::BCEWithLogitsLoss();

	printf("BCE (w Probas): %.4f\n", bce_loss_fn(probas, target).data().item<float>());
	printf("BCE (w Logits): %.4f\n", bce_logits_loss_fn(logits, target).data().item<float>());

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Categorical Cross-entropy\n";
	std::cout << "// -----------------------------------------------------------------\n";
	logits = torch::tensor({{1.5, 0.8, 2.1}});
	probas = torch::softmax(logits, 1);
	target = torch::tensor({2});

	auto cce_loss_fn = torch::nn::NLLLoss();
	auto cce_logits_loss_fn = torch::nn::CrossEntropyLoss();

	printf("CCE (w Logits): %.4f\n", cce_logits_loss_fn(logits, target).data().item<float>());
	printf("CCE (w Probas): %.4f\n", cce_loss_fn(torch::log(probas), target).data().item<float>());

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Implementing a deep convolutional neural network\n";
	std::cout << "// -----------------------------------------------------------------\n";
	torch::nn::Sequential model = torch::nn::Sequential();
	model->push_back("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).padding(2))); //nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
	model->push_back("relu1", torch::nn::ReLU());
	model->push_back("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))); //kernel_size=2))
	model->push_back("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).padding(2))); //nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
	model->push_back("relu2", torch::nn::ReLU());
	model->push_back("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

	x = torch::ones({4, 1, 28, 28});
	std::cout << "model->forward(x).sizes(): " << model->forward(x).sizes() << '\n';

	model->push_back("flatten", torch::nn::Flatten());
	std::cout << "model->forward(x).sizes(): " << model->forward(x).sizes() << '\n';

	model->push_back("fc1", torch::nn::Linear(torch::nn::LinearOptions(3136, 1024)));
	model->push_back("relu3", torch::nn::ReLU());
	model->push_back("dropout", torch::nn::Dropout(0.5));
	model->push_back("fc2", torch::nn::Linear(torch::nn::LinearOptions(1024, 10)));

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

	int num_epochs = 20; // train the training data n times, to save time, we just train 1 epoch
	int batch_size = 100;
	float learning_rate = 0.1;

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
					         std::move(train_dataset), batch_size);

	FASHION valid_datas(FASHION_data_path, FASHION::Mode::kTest);
	auto valid_dataset = valid_datas.map(torch::data::transforms::Stack<>());
	auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
					         std::move(valid_dataset), batch_size);

	model->to(device);

	torch::nn::CrossEntropyLoss loss_fn = torch::nn::CrossEntropyLoss();
	torch::optim::Adam optimizer = torch::optim::Adam(model->parameters(), 0.001);

	std::vector<double> loss_hist_train, accuracy_hist_train,
						loss_hist_valid, accuracy_hist_valid, epoch_steps;

    for(auto& epoch : range(num_epochs, 0)) {
        model->train();
        float loss_train = 0.;
        int accuracy_train = 0;
        int d_size = 0;
        for (auto& batch : *train_loader) {
            torch::Tensor x_batch = batch.data.to(device);
            torch::Tensor y_batch = batch.target.to(device);
            torch::Tensor pred = model->forward(x_batch);
            loss = loss_fn(pred, y_batch);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
            loss_train += loss.data().item<float>()*y_batch.size(0);
            std::optional<long int> dim = {1};
            torch::Tensor is_correct = (torch::argmax(pred, dim) == y_batch).to(torch::kInt32);
            accuracy_train += is_correct.sum().cpu().data().item<int>();
            d_size += y_batch.size(0);
        }

        loss_hist_train.push_back(loss_train/d_size);
        accuracy_hist_train.push_back(accuracy_train*1.0/d_size);
        epoch_steps.push_back((epoch+1)*1.0);

        model->eval();
        torch::NoGradGuard no_grad;
        float loss_valid = 0.;
        int accuracy_valid = 0;
        d_size = 0;
        for(const auto& batch : *valid_loader) {
        	torch::Tensor x_batch = batch.data.to(device);
			torch::Tensor y_batch = batch.target.to(device);
			torch::Tensor pred = model->forward(x_batch);
            loss = loss_fn(pred, y_batch);
            loss_valid += loss.data().item<float>()*y_batch.size(0);
            std::optional<long int> dim = {1};
            torch::Tensor is_correct = (torch::argmax(pred, dim) == y_batch).to(torch::kInt32);
            accuracy_valid += is_correct.sum().cpu().data().item<int>();
            d_size += y_batch.size(0);
        }

        loss_hist_valid.push_back(loss_valid/d_size);
        accuracy_hist_valid.push_back(accuracy_valid*1.0/d_size);

        printf("Epoch %3d train_accuracy: %.3f val_accuracy: %.3f\n", epoch+1, accuracy_hist_train[epoch], accuracy_hist_valid[epoch]);
    }

	auto F = figure(true);
	F->size(1200, 500);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 2);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, epoch_steps, loss_hist_train, "-o")->line_width(2).display_name("Train loss");
	matplot::plot(ax1, epoch_steps, loss_hist_valid, "--<")->line_width(2).display_name("Valid loss");
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "Epoch");
	matplot::ylabel(ax1, "Loss");
	matplot::legend(ax1, {});

	auto ax2 = F->nexttile();
	matplot::hold(ax2, true);
	matplot::plot(ax2, epoch_steps, accuracy_hist_train, "-o")->line_width(2).display_name("Train acc");
	matplot::plot(ax2, epoch_steps, accuracy_hist_valid, "--<")->line_width(2).display_name("Valid acc");
    matplot::hold(ax2, false);
    matplot::xlabel(ax2, "Epoch");
    matplot::ylabel(ax2, "Accuracy");
    matplot::legend(ax2, {});
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}




