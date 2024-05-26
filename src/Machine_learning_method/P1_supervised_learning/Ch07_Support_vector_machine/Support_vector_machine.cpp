/*
 * Support_vector_machine.cpp
 *
 *  Created on: May 4, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>
#include <string>

#include "../../../Algorithms/SupportVectorMachine.h"
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

#include <matplot/matplot.h>
using namespace matplot;



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";

	std::ifstream file;
	std::string path = "./data/breast_cancer.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	//num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> iMap;
    iMap.insert({"malignant", 0});
    iMap.insert({"benign", 1});

    std::cout << "iMap['benign']: " << iMap["benign"] << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(num_records, true);

	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(sidx[i].data().item<int>());
	}

	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, false, true);

	train_dt = train_dt.to(torch::kDouble);
	test_dt = test_dt.to(torch::kDouble);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Support vector machine\n";
	std::cout << "// --------------------------------------------------\n";

	int num_epochs = 1000;
	SVM svm = SVM(train_dt, train_lab);
	torch::Tensor weight = svm.fit(train_dt, train_lab, num_epochs);

	printf("\nScore = %.3f\n", svm.score(test_dt, test_lab, weight, true));

	//torch::Tensor T = torch::tensor({{1,2}, {3,4}});
	//std::cout << T.select(0,1) << '\n';
	std::cout << "Done!\n";
	file.close();
	return 0;
}
