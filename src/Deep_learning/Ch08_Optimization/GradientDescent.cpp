/*
 * GradientDescent.cpp
 *
 *  Created on: Jun 20, 2024
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
#include "../../Utils/helpfunction.h"
#include "../../Utils/csvloader.h"


#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

class GradientDescent {
public:
	GradientDescent(float learning_rate=0.01, int _max_iterations=100) {
        lr = learning_rate;
        max_iterations = _max_iterations;
	}

    torch::Tensor normalization(torch::Tensor X) {
        /*
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        */
    	c10::ArrayRef<long int> dim = {-1};
    	const std::optional<c10::Scalar> p = {2};
    	torch::Tensor l2 = torch::norm(X, p, dim);
        l2.masked_fill_(l2 == 0,  1);
        return X / l2.unsqueeze(1);
    }

    torch::Tensor z_score(torch::Tensor X) {
    	c10::OptionalArrayRef<long int> dim = {0};
    	torch::Tensor mean = torch::mean(X, dim);
        return X.sub(mean) / X.std(0);
    }

    torch::Tensor compute_error(torch::Tensor b, torch::Tensor m, torch::Tensor X, torch::Tensor y) {
    	torch::Tensor total_error = torch::zeros({X.size(0), X.size(0)}, X.dtype()).to(X.device());

        for(auto& i : range(static_cast<int>(X.size(0)), 0)) {
            total_error += torch::pow((y - (torch::mm(m, X.t())) + b), 2);
        }
        return total_error / (1.0*X.size(0));
    }

    std::pair<torch::Tensor, torch::Tensor> step(torch::Tensor b_curr, torch::Tensor m_curr, torch::Tensor X, torch::Tensor y, float learning_rate) {
    	torch::Tensor b_gradient = torch::zeros(b_curr.sizes(), b_curr.dtype()).to(X.device());
    	torch::Tensor m_gradient = torch::zeros(m_curr.sizes(), m_curr.dtype()).to(X.device());
        float N = 1.0*X.size(0);

        for(auto& i : range(static_cast<int>(X.size(0)), 0)) {
        	c10::OptionalArrayRef<long int> dim = {0};
            b_gradient += -(2/N) * torch::sum(y - (torch::mm(X, m_curr.t()) + b_curr), dim);
            c10::OptionalArrayRef<long int> dm = {0};
            m_gradient += -(2/N) * torch::sum(torch::mm(X.t(),  (y - (torch::mm(X, m_curr.t()) + b_curr))), dm);
        }

        torch::Tensor new_b = b_curr - (learning_rate * b_gradient);
        torch::Tensor new_m = m_curr - (learning_rate * m_gradient);
        return std::make_pair(new_b, new_m);
	}

    std::pair<torch::Tensor, torch::Tensor> gradient_descent(torch::Tensor X, torch::Tensor y,
    														 torch::Tensor start_b, torch::Tensor start_m) {
    	torch::Tensor b = start_b.clone();
    	torch::Tensor m = start_m.clone();
        for(auto& i : range(max_iterations, 0)) {
            std::tie(b, m) = step(b, m, X, y, lr);
        }

        return std::make_pair(b, m);
	}
private:
	float lr = 0.;
	int max_iterations;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Load CSV data
	std::ifstream file;
	std::string path = "./data/BostonHousing.csv";
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
	torch::Tensor X, y;
	std::tie(X, y) = process_float_data(file);
	file.close();

	std::cout << "X: " << X.sizes() << '\n';
	std::cout << "y: " << y.sizes() << '\n';

	y = y.unsqueeze(1);
	X = X.to(device);
	y = y.to(device);

	torch::Tensor X_kp = X.clone();
	torch::Tensor y_kp = y.clone();

	torch::Tensor initial_b = torch::tensor({0.0}, torch::kFloat32).to(device);
	torch::Tensor initial_m = torch::zeros({X.size(1), 1}, torch::kFloat32).t().to(device);
    std::cout << initial_m << " " << initial_m.sizes() << '\n';
    torch::nn::init::normal_(initial_m);
    std::cout << initial_m << '\n';

    GradientDescent gd(0.0001, 1000);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  L2 normalized input\n";
	std::cout << "// --------------------------------------------------\n";
    gd.compute_error(initial_b, initial_m, gd.normalization(X), y);

    torch::Tensor bias, slope;
    std::tie(bias, slope) = gd.gradient_descent(gd.normalization(X), y, initial_b, initial_m);
    std::cout << bias.sizes() << " " << slope.sizes() << '\n';

    X = gd.normalization(X);
    std::cout << "y: " << '\n';
    printVector(tensorTovector(y.cpu().squeeze().to(torch::kDouble)));

    torch::Tensor y_pred = (torch::mm(slope, X.t())+bias);
    std::cout << "y_pred: " << '\n';
    printVector(tensorTovector(y_pred.cpu().squeeze().to(torch::kDouble)));

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  z-score normalized input\n";
	std::cout << "// --------------------------------------------------\n";
	X = X_kp;
	y = y_kp;
	initial_b = torch::tensor({0.0}, torch::kFloat32).to(device);
	initial_m = torch::zeros({X.size(1), 1}, torch::kFloat32).t().to(device);
	std::cout << initial_m << " " << initial_m.sizes() << '\n';
	torch::nn::init::normal_(initial_m);
	std::cout << initial_m << '\n';
    gd.compute_error(initial_b, initial_m, gd.z_score(X), y);

    std::tie(bias, slope) = gd.gradient_descent(gd.z_score(X), y, initial_b, initial_m);
    std::cout << bias.sizes() << " " << slope.sizes() << '\n';

    X = gd.z_score(X);
    std::cout << "y: " << '\n';
    printVector(tensorTovector(y.cpu().squeeze().to(torch::kDouble)));

    y_pred = (torch::mm(slope, X.t())+bias);
    std::cout << "y_pred: " << '\n';
    printVector(tensorTovector(y_pred.cpu().squeeze().to(torch::kDouble)));

	std::cout << "Done!\n";
	return 0;
}

