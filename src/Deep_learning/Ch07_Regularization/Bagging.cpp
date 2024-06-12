/*
 * Bagging.cpp
 *
 *  Created on: May 31, 2024
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
#include <limits.h>
#include <ctime>
#include <cstdlib>
#include "../../Algorithms/Bagging.h"
#include "../../Utils/csvloader.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

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

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));

	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> iMap;
	std::cout << "iMap.empty(): " << iMap.empty() << '\n';

	// normalize data = true
	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, false, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";


	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Bagging\n";
	std::cout << "// --------------------------------------------------\n";

	// set random number generator seed
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Set random number generator seed\n";
	std::cout << "// --------------------------------------------------\n";
	std::srand((unsigned) time(NULL));

	int n_estimators = 100, max_depth = 30;
    Bagging model = Bagging(n_estimators, max_depth);
    model.fit(train_dt, train_lab);

    printf("\nAccuracy: %.2f%s\n", model.score( test_dt, test_lab)*100, "%");

	std::cout << "Done!\n";
	return 0;
}




