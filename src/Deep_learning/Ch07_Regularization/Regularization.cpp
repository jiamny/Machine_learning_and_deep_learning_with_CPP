/*
 * Regularization.cpp
 *
 *  Created on: May 21, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/csvloader.h"

using torch::indexing::Slice;
using torch::indexing::None;

class Regularization {
public:
	Regularization(torch::Tensor _X) {
        X = _X;
	}

	torch::Tensor dropout(double drop_probability) {
        /*
        Dropout is a regularization technique for neural networks that drops a unit (along with connections) at
        training time with a specified probability P (a common value is P = 0.5). At test time, all units are present,
        but with weights scaled by p(i.e. w becomes pw ).
        The idea is to prevent co-adaptation, where the neural network becomes too reliant on particular
        connections, as this could be symptomatic of overfitting. Intuitively, dropout can be thought of as creating
         an implicit ensemble of neural networks.
        :param drop_probability: float value between 0 to 1
        */
		double keep_probability;
        if(drop_probability < 1.0) {
            keep_probability = 1 - drop_probability;
        }
        torch::Tensor masker = torch::zeros(X.sizes()).uniform_(0, 1);
        torch::Tensor masked = masker < keep_probability;

        double scale;
        if(keep_probability > 0.0)
            scale = 1.0 / keep_probability;
        else
            scale = 0.0;

        return masked * X * scale;
    }

	torch::Tensor L2_Regularization(torch::Tensor y, torch::Tensor W, double lambda_value) {
        /*
        Weight Decay, or L2 Regularization, is a regularization technique applied to the weights of a neural network.
        We minimize a loss function compromising both the primary loss function and a penalty on the L2 Norm of the
        weights:
                L_new(w) = L_original(w) + lambda * W_T * W
        where  is a value determining the strength of the penalty (encouraging smaller weights).
        Weight decay can be incorporated directly into the weight update rule, rather than just implicitly by defining
        it through to objective function. Often weight decay refers to the implementation where we specify it directly
        in the weight update rule (whereas L2 regularization is usually the implementation which is specified in the
        objective function).
        */
		torch::Tensor Regularization_term = (lambda_value * torch::mm(W, W.t())).to(torch::kDouble) / (2. * y.size(0));
		c10::OptionalArrayRef<long int> dim = {0};
		torch::Tensor output = torch::sum(torch::pow((y - torch::mm(X, W.t())), 2), dim) + Regularization_term;
        return output;
	}

    torch::Tensor L1_Regularization(torch::Tensor y, torch::Tensor W, double lambda_value) {
        /*
         L1 Regularization is a regularization technique applied to the weights of a neural network. We minimize a loss
        function compromising both the primary loss function and a penalty on the L1 Norm of the weights:
            L_new(w) = L_original(w) + lambda * ||W||
        where is a value determining the strength of the penalty. In contrast to weight decay, regularization promotes
        sparsity; i.e. some parameters have an optimal value of zero.
        */
    	c10::OptionalArrayRef<long int> dim = {1};
    	torch::Tensor Regularization_term = torch::sum(
    			(lambda_value * torch::abs(W)).to(torch::kDouble) / (2. * y.size(0)),dim);
    	c10::OptionalArrayRef<long int> dm = {0};
		torch::Tensor output = torch::sum(torch::pow((y - torch::mm(X, W.t())), 2), dm) + Regularization_term;
        return output;
    }
private:
    torch::Tensor X;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Dropout\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor A = torch::arange(20).reshape({5, 4});
	std::cout << "A:\n" << A << '\n';
	Regularization Regularizer(A);
	std::cout << "Regularizer.dropout(0.25):\n" << Regularizer.dropout(0.25) << '\n';
	std::cout << "Regularizer.dropout(0.5):\n" << Regularizer.dropout(0.5) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// L2 Regularization or Weight Decay\n";
	std::cout << "// --------------------------------------------------\n";
	std::ifstream file;
	std::string path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	// Create an unordered_map to hold label names
	std::unordered_map<std::string, int> irisMap;
	irisMap.insert({"Iris-setosa", 0});
	irisMap.insert({"Iris-versicolor", 1});
	irisMap.insert({"Iris-virginica", 2});

	std::cout << "irisMap['Iris-setosa']: " << irisMap["Iris-setosa"] << '\n';
	torch::Tensor X, y;
	std::tie(X, y) = process_data2(file, irisMap, false, false, false);
	file.close();
	X = X.to(torch::kDouble);
	std::cout << "X:\n" << X.sizes() << '\n';
	std::cout << "y:\n" << y.sizes() << '\n';

	torch::Tensor W = torch::zeros(X.size(1), torch::kDouble).uniform_(0, 1).unsqueeze(0);
	std::cout << "X[:10]:\n" << X.index({Slice(0, 10), Slice()}) << '\n';
    Regularization Reg(X);
    torch::Tensor R = Reg.L2_Regularization(y, W, 0.7);
    std::cout << "L2 Regularization:\n" << R << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// L1 Regularization\n";
	std::cout << "// --------------------------------------------------\n";

    W = torch::zeros(X.size(1), torch::kDouble).uniform_(0, 1).unsqueeze(0);
    Regularization Reg1(X);
    std::cout << "X[:10]:\n" << X.index({Slice(0, 10), Slice()}) << '\n';
    torch::Tensor R1 = Reg1.L1_Regularization(y, W, 0.7);
    std::cout << "L1 Regularization:\n" << R1 << '\n';

	std::cout << "Done!\n";
	return 0;
}




