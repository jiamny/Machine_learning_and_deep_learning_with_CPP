/*
 * KNN.cpp
 *
 *  Created on: May 3, 2024
 *      Author: jiamny
 */


#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

#include "../../../Algorithms/KNearestNeighbors.h"
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
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
    std::unordered_map<std::string, int> iMap;
	iMap.insert({"Iris-setosa", 0});
	iMap.insert({"Iris-versicolor", 1});
	iMap.insert({"Iris-virginica", 2});

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
			process_split_data2(file, train_idx, iMap, false, false, false, false); // no normalization

	file.close();

	train_dt = train_dt.to(torch::kDouble);
	test_dt = test_dt.to(torch::kDouble);

	sidx = RangeToensorIndex(train_dt.size(0), true);
	train_dt = torch::index_select(train_dt, 0, sidx.squeeze());
	train_lab = torch::index_select(train_lab, 0, sidx.squeeze());

	sidx = RangeToensorIndex(test_dt.size(0), true);
	test_dt = torch::index_select(test_dt, 0, sidx.squeeze());
	test_lab = torch::index_select(test_lab, 0, sidx.squeeze());

	train_lab = train_lab.reshape({-1});
	test_lab = test_lab.reshape({-1});

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

/*
	std::cout << "train_lab: " << train_lab.sizes() << "\n";
	std::cout << std::get<0>(torch::mode(train_lab, 0, true)) << '\n';
	std::cout << std::get<1>(torch::mode(train_lab, 0, true)) << '\n';

	torch::Tensor dis = torch::tensor({1., 4., 3., 7., 6.});
	torch::Tensor t, tidx;
	tidx = torch::argsort(dis, 0, false);
	std::cout << "t: " << tidx << "\n";
	tidx = tidx.index({Slice(0,2)});
	std::cout << "tidx[:2] " << tidx << "\n";
	std::cout << "dis[tidx] " << dis.index_select(0, tidx) << "\n";

	std::vector<torch::Tensor> tt = {torch::tensor({1}), torch::tensor({3}), torch::tensor({5})};
	std::cout << "tt[0]: " << tt[0].sizes() << '\n';
	std::cout << torch::cat(tt, 0) << '\n' << torch::cat(tt, 0).sizes() << '\n';
	torch::Tensor b = torch::tensor({5.7000, 2.8000, 4.5000, 1.3000});
	torch::Tensor a = torch::tensor({6.7000, 2.5000, 5.8000, 1.8000});

	std::cout << torch::norm(a.sub(b), 2) << '\n';
*/

	KNN knn = KNN(5, train_dt);
	torch::Tensor y_pred = knn.fit_predict(train_dt, train_lab, test_dt);
	std::cout << "train_x: " << train_dt.sizes() << '\n';
	std::cout << "train_y: " << train_lab.sizes() << " <=> " << train_lab.reshape({-1}).sizes()<< '\n';
	std::cout << "y_pred: \n";
	printVector(tensorTovector(y_pred.to(torch::kDouble)));
	printf("\nAccuracy = %.3f\n", knn.accuracy_score(test_lab, y_pred));

	std::cout << "Done!\n";
	return 0;
}
