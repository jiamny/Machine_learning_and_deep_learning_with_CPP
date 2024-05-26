/*
 * AdaBoost.cpp
 *
 *  Created on: May 7, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include "../../../Algorithms/AdaBoost.h"
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
	std::unordered_map<std::string, int> irisMap;
	irisMap.insert({"Iris-setosa", 0});
	irisMap.insert({"Iris-versicolor", 1});
	irisMap.insert({"Iris-virginica", 2});

	std::cout << "irisMap['Iris-setosa']: " << irisMap["Iris-setosa"] << '\n';
	torch::Tensor IX, Iy;
	std::tie(IX, Iy) = process_data2(file, irisMap, false, false, false);
	IX = IX.index({Slice(0, 100), Slice()});
	Iy = Iy.index({Slice(0, 100), Slice()});
	// change setosa => 0 and versicolor + virginica => 1
	Iy.masked_fill_(Iy < 1, -1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(100, true);
	printVector(tensorTovector(sidx.squeeze().to(torch::kDouble)));

	IX = torch::index_select(IX, 0, sidx.squeeze());
	Iy = torch::index_select(Iy, 0, sidx.squeeze());

	torch::Tensor X_train = IX.index({Slice(0, 70), Slice()});
	torch::Tensor X_test = IX.index({Slice(70, None), Slice()});
	torch::Tensor y_train = Iy.index({Slice(0, 70), Slice()});
	torch::Tensor y_test = Iy.index({Slice(70, None), Slice()});
	std::cout << "Train size = " << X_train.sizes() << "\n";
	std::cout << "Test size = " << X_test.sizes() << "\n";
	y_train.squeeze_();
	y_test.squeeze_();

	/*
	torch::Tensor w = torch::zeros({5}, torch::kFloat32);
	w.fill_(1.0/200);
	std::cout << "w: " << w << '\n';

	torch::Tensor X = torch::tensor({{0, 1, 3}, {0, 3, 1}, {1, 2, 2}, {1, 1, 3}, {1, 2, 3}, {0, 1, 2},
	              {1, 1, 2}, {1, 1, 1}, {1, 3, 1}, {0, 2, 1}}, torch::kDouble);
	torch::Tensor y = torch::tensor({-1, -1, -1, -1, -1, -1, 1, 1, -1, -1}, torch::kInt32);
	*/
	AdaBoost clf = AdaBoost(X_train, y_train);
	clf.fit();
	torch::Tensor y_predict = clf.predict(X_test);
	double score = clf.score(X_test, y_test);
	std::cout << "原始输出:\n";
	printVector(tensorTovector(y_test.squeeze().to(torch::kDouble)));
	std::cout << "预测输出:\n";
	printVector(tensorTovector(y_predict.to(torch::kDouble)));

	printf("预测正确率： %3.1f%s\n", score*100, "%");
	std::cout << "Done!\n";
	return 0;
}



