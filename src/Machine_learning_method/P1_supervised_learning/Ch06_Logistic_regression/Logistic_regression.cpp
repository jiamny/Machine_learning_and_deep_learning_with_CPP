/*
 * Logistic_regression.cpp
 *
 *  Created on: Apr 25, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include "../../../Algorithms/LogisticRegression.h"
#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
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

	// change setosa => 0 and versicolor + virginica => 1
	y.masked_fill_(y > 0, 1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(num_records, true);

	X = torch::index_select(X, 0, sidx.squeeze());
	y = torch::index_select(y, 0, sidx.squeeze());

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Logistic regression\n";
	std::cout << "// --------------------------------------------------\n";

    auto lr = LogisticRegression(X);
    torch::Tensor y_predict, w, b;
    std::tie(w, b) = lr.run(X, y);
    y_predict = lr.predict(X);
    y_predict = y_predict.to(y.dtype());

    torch::Tensor yhat_y = torch::cat({y_predict, y}, 1);
	std::cout << "yhat_y:\n" <<   yhat_y << '\n';

    auto cmp = (y_predict == y );
    auto ac = (torch::sum(cmp.to(y.dtype()))*1.0 / X.size(0)) *100;
    std::cout << "Accuracy: " << ac.data().item<float>() << '\n';

	std::cout << "Done!\n";
	file.close();
	return 0;
}




