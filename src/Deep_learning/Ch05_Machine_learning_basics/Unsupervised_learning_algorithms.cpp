/*
 * Unsupervised_learning_algorithms.cpp
 *
 *  Created on: May 1, 2024
 *      Author: jiamny
 */
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <cmath>
#include "../../Algorithms/PrincipalComponentsAnalysis.h"
#include "../../Algorithms/Kmeans_clustering.h"
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
	std::cout << "// 主成分分析法; Principal Components Analysis\n";
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
	// # 查看数据
    std::cout << "data[0]:\n" << X[0] << '\n';
    std::cout << "label[0]:\n" << y[0] << '\n';

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

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// k-均值聚类; k-means\n";
	std::cout << "// --------------------------------------------------\n";
	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);
    torch::Tensor IX, Iy;
    std::tie(IX, Iy) = process_data2(file, irisMap, false, false, false);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);
	printVector(tensorTovector(sidx.squeeze().to(torch::kDouble)));

	IX = torch::index_select(IX, 0, sidx.squeeze());
	Iy = torch::index_select(Iy, 0, sidx.squeeze());
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// select using first two features\n";
	std::cout << "// --------------------------------------------------\n";

	IX = IX.index({Slice(), Slice(0, 2)});
	std::cout << "X size = " << IX.sizes() << "\n";

	int64_t n_classes = 3;
	int64_t iterations=3000;
	std::string method = "euclidean";
	KMeans kmeans = KMeans(IX, n_classes, iterations, method);
	kmeans.fit(IX);
	torch::Tensor labels = kmeans.predict(IX);
	std::cout << "labels: " << labels.sizes() << "\n";
	printVector(tensorTovector(labels.to(torch::kDouble)));

	torch::Tensor centers = kmeans.KMeans_Centroids;
	std::vector<double> cnt_0, cnt_1, cnt_2, cntx, cnty;
	std::vector<int> cntC;
	for( int i = 0; i < centers.size(0); i++) {
		if( i == 0 ) {
			cnt_0 = tensorTovector(centers.select(0, 0).to(torch::kDouble));
			cntx.push_back(cnt_0[0]);
			cnty.push_back(cnt_0[1]);
		}
		if( i == 1 ) {
			cnt_1 = tensorTovector(centers.select(0, 1).to(torch::kDouble));
			cntx.push_back(cnt_1[0]);
			cnty.push_back(cnt_1[1]);
		}

		if( i == 2 ) {
			cnt_2 = tensorTovector(centers.select(0, 2).to(torch::kDouble));
			cntx.push_back(cnt_2[0]);
			cnty.push_back(cnt_2[1]);
		}
		 cntC.push_back(i);
	}

/*
    std::vector<int64_t> idx_0, idx_1, idx_2;
    for(int64_t i = 0; i < labels.size(0); i++) {
     	int l = labels[i].data().item<int>();
     	if( l == 0 ) idx_0.push_back(i);
     	if( l == 1 ) idx_1.push_back(i);
     	if( l == 2 ) idx_2.push_back(i);
    }

    torch::Tensor index_0 = torch::from_blob(idx_0.data(), {static_cast<int>(idx_0.size())}, torch::kLong).clone();
    torch::Tensor index_1 = torch::from_blob(idx_1.data(), {static_cast<int>(idx_1.size())}, torch::kLong).clone();
    torch::Tensor index_2 = torch::from_blob(idx_2.data(), {static_cast<int>(idx_2.size())}, torch::kLong).clone();
*/
    auto F = figure(true);
    F->size(900, 700);
    F->add_axes(false);
    F->reactive_mode(false);
    F->tiledlayout(1, 1);
    F->position(0, 0);

    double fsz = 12;
    std::vector<double> fxx, fyy;
    std::vector<int> fc;
    for(int64_t i = 0; i < labels.size(0); i++) {
      	fxx.push_back(IX.index({i, 0}).data().item<double>());
      	fyy.push_back(IX.index({i, 1}).data().item<double>());
      	fc.push_back(labels[i].data().item<int>());
    }

    auto fax = F->nexttile();
    auto fst = scatter(fax, fxx, fyy, fsz, fc);
    fst->marker_face(true);
    fax->xlabel("sepal length");
    fax->ylabel("sepal width");
    fax->grid(on);
    fax->hold(on);
    auto fct = scatter(fax, cntx, cnty, 25, cntC);
    fct->marker_style(line_spec::marker_style::upward_pointing_triangle).marker_face(true);
    //fct->marker_face(true);
    //c10::OptionalArrayRef<long int> dim = {0};
    //torch::Tensor xy_0 = torch::mean(torch::index_select(transformed, 0, index_0), dim);
    //torch::Tensor xy_1 = torch::mean(torch::index_select(transformed, 0, index_1), dim);
    //torch::Tensor xy_2 = torch::mean(torch::index_select(transformed, 0, index_2), dim);
    fax->text(cnt_0[0], cnt_0[1], "← setosa")->font_size(14);
    fax->text(cnt_1[0], cnt_1[1], "← versicolor")->font_size(14);
    fax->text(cnt_2[0], cnt_2[1], "← virginica")->font_size(14);
    show();

	std::cout << "Done!\n";
	file.close();
	return 0;
}

