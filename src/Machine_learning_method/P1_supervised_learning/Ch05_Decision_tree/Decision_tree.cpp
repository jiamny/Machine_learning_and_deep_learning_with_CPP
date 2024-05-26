/*
 * Decision_tree.cpp
 *
 *  Created on: Apr 26, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include "../../../Algorithms/Decision_tree.h"
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";

	std::ifstream file;
	std::string path = "./data/breast_cancer_wisconsin.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records -= 1; 		// if first row is heads but not record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	//std::vector<int> indices;
	//for( int i = 0; i < num_records; i++ ) indices.push_back(i);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
	torch::Tensor sidx = RangeToensorIndex(num_records, true);

	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));
	printVector(indices);
	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> iMap;
	std::cout << "iMap.empty(): " << iMap.empty() << '\n';

	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, false, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	auto m = torch::tensor({1, 5, 2, 3, 3, 5, 5});
	std::cout << std::get<0>(at::_unique(m)) << '\n';
	std::cout << std::get<0>(at::_unique(m)).size(0) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Decision tree\n";
	std::cout << "// --------------------------------------------------\n";

	int max_depth = 10;
	DecisionTree_CART classifier = DecisionTree_CART(max_depth);
    classifier.fit(train_dt, train_lab);

    std::vector<int> y_predict = classifier.predict(test_dt);

    torch::Tensor pred = torch::from_blob(
    		y_predict.data(), {static_cast<long int>(y_predict.size())}, at::TensorOptions(torch::kInt)).clone();

    int correct = torch::sum((test_lab.squeeze() == pred).to(torch::kInt)).data().item<int>();

    std::cout << "Accuracy: " << (correct*1.0/y_predict.size())*100 << "%\n";

    classifier.print_tree();

	std::cout << "Done!\n";
	file.close();

	return 0;
}


