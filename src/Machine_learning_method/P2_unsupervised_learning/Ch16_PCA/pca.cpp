/*
 * pca.cpp
 *
 *  Created on: Apr 23, 2024
 *      Author: jiamny
 */


#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <matplot/matplot.h>
#include "../../../Algorithms/PrincipalComponentsAnalysis.h"
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;
using namespace matplot;

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Device device = torch::Device(torch::kCPU);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";

	// # 载入数据
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
	std::cout << "iris records in file = " << num_records << '\n';

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

    int Iris_setosa = 0, Iris_versicolor = 0, Iris_virginica = 0;
    for(int64_t i = 0; i < y.size(0); i++) {
    	if( y[i].data().item<int>() == 0)
    		Iris_setosa += 1;
    	if( y[i].data().item<int>() == 1)
    		Iris_versicolor += 1;
    	if( y[i].data().item<int>() == 2)
    		Iris_virginica += 1;
    }
    std::cout << "Iris_setosa: " << Iris_setosa << '\n';
    std::cout << "Iris_versicolor: " << Iris_versicolor << '\n';
    std::cout << "Iris_virginica: " << Iris_virginica << '\n';

    // # 查看数据
    printf("%20s %20s %20s %20s %10s\n", "sepal length","sepal width", "petal length", "petal width", "label");
    for(int64_t i = 0; i < 5; i++) {
    	printf("%18.1f %18.1f %18.1f %18.1f %20d\n", X.index({i,0}).data().item<float>(),
    			X.index({i,1}).data().item<float>(), X.index({i,2}).data().item<float>(),
				X.index({i,3}).data().item<float>(), y[i].data().item<int>());
    }

	// # 查看数据
    std::cout << "data[0]:\n" << X[0] << '\n';
    std::cout << "label[0]:\n" << y[0] << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  PCA \n";
	std::cout << "// --------------------------------------------------\n";

    int64_t n_components = 2;
    auto pca = PCA(n_components);
	torch::Tensor transformed = pca.fit_transform(X);

    std::vector<int64_t> idx_0, idx_1, idx_2;
    for(int64_t i = 0; i < y.size(0); i++) {
     	int l = y[i].data().item<int>();
     	if( l == 0 ) idx_0.push_back(i);
     	if( l == 1 ) idx_1.push_back(i);
     	if( l == 2 ) idx_2.push_back(i);
    }

    torch::Tensor index_0 = torch::from_blob(idx_0.data(), {static_cast<int>(idx_0.size())}, torch::kLong).clone();
    torch::Tensor index_1 = torch::from_blob(idx_1.data(), {static_cast<int>(idx_1.size())}, torch::kLong).clone();
    torch::Tensor index_2 = torch::from_blob(idx_2.data(), {static_cast<int>(idx_2.size())}, torch::kLong).clone();

    auto h = figure(true);
    h->size(800, 600);
    h->add_axes(false);
    h->reactive_mode(false);
    h->tiledlayout(1, 1);
    h->position(0, 0);

    double sz = 12;
    std::vector<double> xx, yy;
    std::vector<int> c;
    for(int64_t i = 0; i < y.size(0); i++) {
      	xx.push_back(transformed.index({i, 0}).data().item<double>());
      	yy.push_back(transformed.index({i, 1}).data().item<double>());
      	c.push_back(y[i].data().item<int>());
    }
    auto ax1 = h->nexttile();
    auto st = scatter(ax1, xx, yy, sz, c);
    st->marker_face(true);
    ax1->xlabel("PC 1");
    ax1->ylabel("PC 2");
    ax1->grid(on);
    c10::OptionalArrayRef<long int> dim = {0};
    torch::Tensor xy_0 = torch::mean(torch::index_select(transformed, 0, index_0), dim);
    torch::Tensor xy_1 = torch::mean(torch::index_select(transformed, 0, index_1), dim);
    torch::Tensor xy_2 = torch::mean(torch::index_select(transformed, 0, index_2), dim);
    ax1->text(xy_0[0].data().item<double>(), xy_0[1].data().item<double>(), "← setosa")->font_size(12);
    ax1->text(xy_1[0].data().item<double>(), xy_1[1].data().item<double>(), "← versicolor")->font_size(12);
    ax1->text(xy_2[0].data().item<double>(), xy_2[1].data().item<double>(), "← virginica")->font_size(12);
    show();

	std::cout << "Done\n";
	file.close();
	return 0;
}
