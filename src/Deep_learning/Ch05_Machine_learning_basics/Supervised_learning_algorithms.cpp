/*
 * Supervised_learning_algorithms.cpp
 *
 *  Created on: May 1, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <cmath>
#include "../../Algorithms/Decision_tree.h"
#include "../../Algorithms/LogisticRegression.h"
#include "../../Algorithms/SupportVectorMachine.h"
#include "../../Algorithms/KNearestNeighbors.h"
#include "../../Utils/csvloader.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 逻辑回归; LogisticRegression\n";
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
	torch::Tensor IX, Iy;
	std::tie(IX, Iy) = process_data2(file, irisMap, false, false, false);
	IX = IX.index({Slice(0, 100), Slice()});
	Iy = Iy.index({Slice(0, 100), Slice()});
	// change setosa => 0 and versicolor + virginica => 1
	Iy.masked_fill_(Iy > 0, 1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(100, true);
	printVector(tensorTovector(sidx.squeeze().to(torch::kDouble)));

	IX = torch::index_select(IX, 0, sidx.squeeze());
	Iy = torch::index_select(Iy, 0, sidx.squeeze());

	torch::Tensor X_train = IX.index({Slice(0, 70), Slice()});
	torch::Tensor X_test = IX.index({Slice(70, None), Slice()});
	torch::Tensor y_train = Iy.index({Slice(0, 70), Slice()});
	torch::Tensor y_test = Iy.index({Slice(70, None), Slice()});
	std::cout << "Train size = " << X_train.sizes() << "\n";
	std::cout << "Test size = " << X_test.sizes() << "\n";

	LogisticRegression LR = LogisticRegression(X_train);
	torch::Tensor w, b;
	std::tie(w, b) = LR.run(X_train, y_train);
	printf("\nLogisticRegression Score = %3.1f%s\n", LR.score(X_test, y_test, false)*100, "%");
	file.close();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// ⽀持向量机; Support Vector Machine\n";
	std::cout << "// --------------------------------------------------\n";
	path = "./data/breast_cancer.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}
	num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
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
	sidx = RangeTensorIndex(num_records, true);

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

	int num_epochs = 1000;
	SVM svm = SVM(train_dt, train_lab);
	torch::Tensor weight = svm.fit(train_dt, train_lab, num_epochs);

	printf("\nSupport Vector Machine Score = %.3f%s\n", svm.score(test_dt, test_lab, weight, false)*100, "%");
	file.close();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// k-近邻; k-Nearest Neighbors\n";
	std::cout << "// --------------------------------------------------\n";
	path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	sidx = RangeTensorIndex(num_records, true);

	// ---- split train and test datasets
	std::unordered_set<int> ktrain_idx;
	train_size = static_cast<int>(num_records * 0.80);
	std::cout << "train_sz: " << train_size << '\n';
	for( int i = 0; i < train_size; i++ ) {
		ktrain_idx.insert(sidx[i].data().item<int>());
	}

	torch::Tensor ktrain_dt, ktrain_lab, ktest_dt, ktest_lab;
	std::tie(ktrain_dt, ktrain_lab, ktest_dt, ktest_lab) =
				process_split_data2(file, ktrain_idx, iMap, false, false, false, false); // no normalization

	file.close();
	std::cout << "ktrain_dt: " << ktrain_dt.sizes() << '\n';

	ktrain_dt = ktrain_dt.to(torch::kDouble);
	ktest_dt = ktest_dt.to(torch::kDouble);

	sidx = RangeTensorIndex(ktrain_dt.size(0), true);
	ktrain_dt = torch::index_select(ktrain_dt, 0, sidx.squeeze());
	ktrain_lab = torch::index_select(ktrain_lab, 0, sidx.squeeze());

	sidx = RangeTensorIndex(ktest_dt.size(0), true);
	ktest_dt = torch::index_select(ktest_dt, 0, sidx.squeeze());
	ktest_lab = torch::index_select(ktest_lab, 0, sidx.squeeze());

	ktrain_lab = ktrain_lab.reshape({-1});
	ktest_lab = ktest_lab.reshape({-1});

	std::cout << "Train size = " << ktrain_dt.sizes() << "\n";
	std::cout << "Test size = " << ktest_dt.sizes() << "\n";

	KNN knn = KNN(5, train_dt);
	torch::Tensor y_pred = knn.fit_predict(train_dt, train_lab, test_dt);
	printf("\nk-Nearest Neighbors accuracy = %3.1f%s\n", knn.accuracy_score(test_lab, y_pred)*100, "%");

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 决策树; Decision Tree\n";
	std::cout << "// --------------------------------------------------\n";
	path = "./data/breast_cancer_wisconsin.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	sidx = RangeTensorIndex(num_records -1, true);
	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));

	// ---- split train and test datasets
	train_idx.clear();
	train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> jMap;
	//torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, jMap, false, false, false, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	int max_depth = 5;
	DecisionTree_CART classifier = DecisionTree_CART(max_depth);
    classifier.fit(train_dt, train_lab);
    printf("\nDecision Tree Score = %3.1f%s\n", classifier.score(test_dt, test_lab, false)*100, "%");
    file.close();

	std::cout << "Done!\n";
	return 0;
}



