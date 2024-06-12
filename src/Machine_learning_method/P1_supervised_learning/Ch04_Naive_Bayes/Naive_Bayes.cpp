/*
 * Naive_Bayes.cpp
 *
 *  Created on: May 1, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include "../../../Algorithms/NaiveBayes.h"
#include "../../../Utils/csvloader.h"


using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";
	std::ifstream file;
	std::string path = "./data/wine.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n'); 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';
	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    torch::Tensor X, y;
    std::tie(X, y) =  load_label_fst_data(file, true, true);
	std::cout << "X size = " << X.sizes() << "\n";
	std::cout << "y size = " << y.sizes() << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);
	printVector(tensorTovector(sidx.squeeze().to(torch::kDouble)));

	X = torch::index_select(X, 0, sidx.squeeze());
	y = torch::index_select(y, 0, sidx.squeeze());

	int num_train = static_cast<int>(num_records * 0.7);

	torch::Tensor X_train = X.index({Slice(0, num_train), Slice()});
	torch::Tensor y_train = y.index({Slice(0, num_train), Slice()});
	torch::Tensor X_test  = X.index({Slice(num_train, None), Slice()});
	torch::Tensor y_test  = y.index({Slice(num_train, None), Slice()});

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Naive Bayes\n";
	std::cout << "// --------------------------------------------------\n";

	NaiveBayes model = NaiveBayes();
	model.fit(X_train, y_train);
	printf("\nScore = %.3f\n", model.score(X_test, y_test, true));

	std::cout << "Done!\n";
	file.close();
	return 0;
}



