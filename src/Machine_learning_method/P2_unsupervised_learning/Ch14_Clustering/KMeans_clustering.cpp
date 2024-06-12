/*
 * KMeans_clustering.cpp
 *
 *  Created on: Apr 29, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

#include "../../../Algorithms/Kmeans_clustering.h"
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

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.90);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(sidx[i].data().item<int>());
	}

	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, irisMap, false, false, false, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  KMeans clustering\n";
	std::cout << "// --------------------------------------------------\n";

	int64_t n_classes = 3;
	int64_t iterations=3000;

	std::string method = "euclidean"; //, "manhattan", "cosine";
	float max_ac = 10.0;
	torch::Tensor KMeans_Centroids;

	for(auto& r : range(200)) {
		KMeans kmeans = KMeans(train_dt, n_classes, iterations, method);
		//kmeans.initialize_centroid(train_dt, n_classes);
		kmeans.fit(train_dt);

		torch::Tensor ypred = kmeans.predict(test_dt);

		ypred = ypred.to(test_lab.dtype());

		auto cmp = (ypred == test_lab.squeeze() );
		auto ac = (torch::sum(cmp.to(test_lab.dtype()))*1.0 / test_dt.size(0)) *100;
		if( ac.data().item<float>() > max_ac ) {
			max_ac = ac.data().item<float>();
			KMeans_Centroids = kmeans.KMeans_Centroids.clone();
		}
		if( r % 20 == 0 )
			printf( "%03d run; ac = %.03f\n", r, max_ac);
	}

	if( max_ac > 50 ) {
		std::cout << "kmeans.KMeans_Centroids: " << KMeans_Centroids << '\n';
		std::cout << "Accuracy with "  << method << ": " << max_ac << "%\n";
	}
/*
    torch::Tensor a = torch::tensor({{1,2,3}, {4,5,6}, {7,8,9}}).to(torch::kDouble);
    torch::Tensor cd = torch::zeros({4,3}).to(a.dtype());
    std::cout << "cd-1:\n" << cd << '\n';
    for(int64_t j = 0; j < 4; j++) {
    	c10::OptionalArrayRef<long int> didx = {0};
    	torch::Tensor mn = a.mean(didx);
    	cd.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({j})}, mn);
    }
    std::cout << "cd-2:\n" << cd << '\n';
*/
	std::cout << "Done!\n";
	file.close();
	return 0;
}


