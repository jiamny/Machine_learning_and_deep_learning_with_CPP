/*
 * BatchNormalization.cpp
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
#include <matplot/matplot.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/activation.h"


using torch::indexing::Slice;
using torch::indexing::None;

using namespace matplot;

struct Net : public torch::nn::Module {
	bool do_bn = false;
	torch::nn::BatchNorm1d bn_input{nullptr};
	int N_HIDDEN = 0;
	std::vector<torch::nn::Linear> fcs;
	std::vector<torch::nn::BatchNorm1d> bns;
	torch::nn::Linear predict{nullptr};
	double B_INIT = -0.2;   // use a bad bias constant initializer

    Net(int _N_HIDDEN, bool batch_normalization=false) {
        do_bn = batch_normalization;
        N_HIDDEN = _N_HIDDEN;

        bn_input = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1).momentum(0.5)); // for input data

        for(auto& i : range(N_HIDDEN, 0)) {               // build hidden layers and BN layers
            int input_size = 1;
            if( i != 0 )
            	input_size = 10;
            torch::nn::Linear fc = torch::nn::Linear(torch::nn::LinearOptions(input_size, 10));
            register_module("fc"+std::to_string(i), fc);       // IMPORTANT set layer to the Module
            _set_init(fc);                  // parameters initialization
            fcs.push_back(fc);
            if(do_bn) {
            	torch::nn::BatchNorm1d bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(10).momentum(0.5));
            	register_module("bn"+std::to_string(i), bn);   // IMPORTANT set layer to the Module
                bns.push_back(bn);
            }
        }

        predict = torch::nn::Linear(torch::nn::LinearOptions(10, 1));         // output layer
        register_module("predict", predict);
        _set_init(predict);            // parameters initialization
    }

    void _set_init(torch::nn::Linear &m) {
    	torch::nn::init::normal_(m.get()->weight, 0., .1);
    	torch::nn::init::constant_(m.get()->bias, B_INIT);
    }

    std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(torch::Tensor x) {
    	std::vector<torch::Tensor> pre_activation = {x.clone()};

    	if( do_bn ) {
        	x = bn_input->forward(x);     // input batch normalization
        }

        std::vector<torch::Tensor> layer_input = {x.clone()};

        for(auto& i : range(N_HIDDEN, 0)) {

            x = fcs[i]->forward(x);
            pre_activation.push_back(x.clone());

            if(do_bn ) {
            	x = bns[i]->forward(x);   // batch normalization
            }

            x = torch::tanh(x);
            layer_input.push_back(x.clone());
        }
        torch::Tensor out = predict(x);

        return std::make_tuple(out, layer_input, pre_activation);
    }
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Hyper parameters
	int N_SAMPLES = 2000;
	int BATCH_SIZE = 64;
	int EPOCH = 12;
	double LR = 0.03;
	int N_HIDDEN = 8;

	// training data
	torch::Tensor x = torch::linspace(-7, 10, N_SAMPLES).reshape({N_SAMPLES, 1}).to(torch::kFloat32);

	torch::Tensor noise = torch::normal(0, 2, x.sizes()).to(torch::kFloat32);
	torch::Tensor y = torch::square(x) - 5 + noise;

	// test data
	torch::Tensor test_x = torch::linspace(-7, 10, 200).reshape({200, 1}).to(torch::kFloat32);
	noise = torch::normal(0, 2, test_x.sizes()).to(torch::kFloat32);
	torch::Tensor test_y = torch::square(test_x) - 5 + noise;

	auto train_dataset = LRdataset(x, y)
		.map(torch::data::transforms::Stack<>());

	auto train_loader =
		      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		    		  std::move(train_dataset), BATCH_SIZE);

	std::vector<Net> nets = {Net(N_HIDDEN, false), Net(N_HIDDEN, true)};

	std::cout << nets[0] << '\n';
	std::cout << nets[1] << '\n';

	auto opt0 = torch::optim::Adam(nets[0].parameters(), LR);
	auto opt1 = torch::optim::Adam(nets[1].parameters(), LR);

	auto loss_func = torch::nn::MSELoss();

	std::vector<double> d1, d2;
	std::vector<std::vector<double>> losses = {d1, d2};
	std::vector<double> steps;

	for(auto& epoch : range(EPOCH, 0)) {

		std::vector<std::vector<torch::Tensor>> layer_inputs, pre_acts;

		for(int i = 0; i < nets.size(); i++) {
			Net net = nets[i];
			net.eval();				// set eval mode to fix moving_mean and moving_var
			torch::Tensor pred;
			std::vector<torch::Tensor> layer_input, pre_act;
			std::tie(pred, layer_input, pre_act) = net.forward(test_x);
			losses[i].push_back(loss_func(pred, test_y).data().item<float>());
			layer_inputs.push_back(layer_input);
			pre_acts.push_back(pre_act);
			net.train();             // free moving_mean and moving_var
		}

		for(int i = 0; i < nets.size(); i++) {
			Net net = nets[i];
			torch::Tensor pred;
			std::vector<torch::Tensor> lin, pac;
			net.train();
			for(auto& batch : *train_loader) {
				auto data = batch.data;
				auto targets = batch.target;

				std::tie(pred, lin, pac) = net.forward(data);
				auto loss = loss_func(pred, targets);
				if( i == 0 ) {
					opt0.zero_grad();
					loss.backward();
					opt0.step();
				} else {
					opt1.zero_grad();
					loss.backward();
					opt1.step();
				}
			}
		}
		steps.push_back((epoch+1)*1.0);
		std::cout << "Epoch: " << (epoch + 1) << " / " << EPOCH << '\n';
	}
	printVector(losses[0]);
	printVector(losses[1]);

	std::vector<std::vector<double>> preds;
	for(auto& net : nets) {    //set eval mode to fix moving_mean and moving_var
		net.eval();
		torch::Tensor rlt = std::get<0>(net.forward(test_x));
		preds.push_back(tensorTovector(rlt.to(torch::kDouble)));
	}

	auto F = figure(true);
	F->size(1500, 450);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	tiledlayout(1, 3);

	auto ax1 = nexttile();
	double sz = 10;

	auto l = scatter(ax1, tensorTovector(x.to(torch::kDouble)), tensorTovector(y.to(torch::kDouble)), sz);
	l->marker_color({1.f, 0.f, 0.f});
	l->marker_style(line_spec::marker_style::circle);
	title(ax1, "train data");

	auto ax2 = nexttile();
	plot(ax2, steps, losses[0], "m-")->line_width(3).display_name("No Batch Normalization");
	hold(ax2, on);
	plot(ax2, steps, losses[1], "r-.")->line_width(3).display_name("Batch Normalization");
	xlabel(ax2, "epoch");
	ylabel(ax2, "test_loss");
	ylim(ax2, {0, 2000});
	legend({});
	title(ax2, "Without vs. with batch normalization");

	auto ax3 = nexttile();
	plot(ax3, tensorTovector(test_x.to(torch::kDouble)), preds[0], "m-")->line_width(4).display_name("No Batch Normalization");
	hold(ax3, on);
	plot(ax3, tensorTovector(test_x.to(torch::kDouble)), preds[1], "r-.")->line_width(4).display_name("Batch Normalization");
	scatter(ax3, tensorTovector(test_x.to(torch::kDouble)),
			tensorTovector(test_y.to(torch::kDouble)), sz)->display_name("test data");
	legend(ax3, {});
	title(ax3, "evaluation");
	F->draw();

	show();

	std::cout << "Done!\n";
	return 0;
}




