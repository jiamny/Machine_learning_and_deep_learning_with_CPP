/*
 * ElasticNetRegression.cpp
 *
 *  Created on: Aug 29, 2024
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

#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

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

class ElasticNetRegression {

public:
	ElasticNetRegression(double _learning_rate, int _max_iterations, double _l1_penality, double _l2_penality) {
        lr = _learning_rate;
        max_iterations = _max_iterations;
        l1_penality = _l1_penality;
        l2_penality = _l2_penality;
	}

    void fit(torch::Tensor _X, torch::Tensor _y) {
        m = _X.size(0);
        n = _X.size(1);

        w = torch::zeros({n, 1}, torch::kDouble);
        b = 0.0;
        X = _X;
        y = _y;

        for( auto& i : range(max_iterations, 0)) {
            update_weights();
        }
    }

    void update_weights() {
    	torch::Tensor y_pred = predict(X);
    	torch::Tensor dw = torch::zeros({n, 1}).to(torch::kDouble);

        for(auto& j : range(n, 0)) {

            if( w[j].data().item<double>() > 0 ) {
                dw[j] = ( - (2* torch::mm(X.index({Slice(), j}).unsqueeze(0), (y - y_pred))
                		  + l1_penality + 2 * l2_penality * w[j])).data().item<double>() / m;
            } else {
                dw[j] = (-(2 * torch::mm(X.index({Slice(), j}).unsqueeze(0), (y - y_pred))
                         - l1_penality + 2 * l2_penality * w[j])).data().item<double>() / m;
            }
        }

        torch::Tensor  db = -2 * torch::sum(y - y_pred) / m;
        w = w - lr * dw;
        b = b - lr * db.data().item<double>();
    }

    torch::Tensor predict(torch::Tensor X) {
        return torch::mm(X, w) + b;
    }

    double get_b() {
    	return b;
    }

    torch::Tensor get_w() {
        	return w;
    }

private:
	double lr = 0., l1_penality = 0., l2_penality = 0., b = 0.;
	int max_iterations = 0, m = 0, n = 0;
	torch::Tensor w, X, y;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load data\n";
	std::cout << "// --------------------------------------------------\n";

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

	y = y.reshape({-1, 1});
	X = X.to(torch::kDouble);
	y = y.to(torch::kDouble);

	std::cout << "X: " << X.sizes() << '\n'
			  << X.index({Slice(0,10), Slice()}) << '\n';
	std::cout << "y: " << y.sizes() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Normalization data\n";
	std::cout << "// --------------------------------------------------\n";

	X = normalization(X);
	std::cout << "X: " << X.sizes() << '\n'
				  << X.index({Slice(0,10), Slice()}) << '\n';

    torch::Tensor x_train, x_test, y_train, y_test;
    std::tie(x_train, x_test, y_train, y_test) = train_test_split(X, y, 0.3, true);
    std::cout << x_train.sizes() << '\n' << y_train.sizes() << '\n';

    int max_iterations = 1000;
    double learning_rate = 0.001, l1_penality = 500, l2_penality = 1;
    ElasticNetRegression regression(learning_rate, max_iterations, l1_penality, l2_penality);

    regression.fit(x_train, y_train);
    torch::Tensor Y_pred = regression.predict(x_test);
    std::cout << "Predicted values: " << Y_pred.index({Slice(0, 3)}) << '\n';
    std::cout << "Real values: " << y_test.index({Slice(0, 3)}) << '\n';
    std::cout << "Trained W: " << regression.get_w()[0] << '\n';
	std::cout << "Trained b: " << regression.get_b() << '\n';


	std::cout << "Done!\n";
	return 0;
}




