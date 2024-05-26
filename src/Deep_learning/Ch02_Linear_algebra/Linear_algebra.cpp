/*
 * Linear_algebra.cpp
 *
 *  Created on: Apr 21, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <matplot/matplot.h>
#include "../../Utils/csvloader.h"
#include "../../Utils/TempHelpFunctions.h"

using namespace torch::autograd;
using torch::indexing::Slice;
using torch::indexing::None;
using namespace matplot;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Device device = torch::Device(torch::kCPU);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 标量, 向量, 矩阵, 张量; Scalars, Vectors, Matrices and Tensors\n";
	std::cout << "// --------------------------------------------------\n";
	//"# 标量";
	int s = 5;

	// "# 向量";
	auto v = torch::tensor({1,2});

	//"# 矩阵";
	auto m = torch::tensor({{1,2}, {3,4}});

	//"# 张量";
	auto t = torch::tensor({
		{{1,2,3},{4,5,6},{7,8,9}},
		{{11,12,13},{14,15,16},{17,18,19}},
		{{21,22,23},{24,25,26},{27,28,29}},
	});

	std::cout << "标量: " + std::to_string(s) << '\n';
	std::cout << "\n向量:\n" << v << '\n';
	std::cout << "\n矩阵:\n" << m << '\n';
	std::cout << "\n张量:\n" << t << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 矩阵转置; Transpose\n";
	std::cout << "// --------------------------------------------------\n\n";
	auto A = torch::tensor({{1.0,2.0},{1.0,0.0},{2.0,3.0}});
	auto A_t = A.t();
	std::cout << "A:\n" << A << '\n';
	std::cout << "A 的转置:\n" << A_t << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 矩阵加法; matrix addition\n";
	std::cout << "// --------------------------------------------------\n\n";
	auto a = torch::tensor({{1.0,2.0},{3.0,4.0}});
	auto b = torch::tensor({{6.0,7.0},{8.0,9.0}});
	std::cout << "矩阵 a：\n" << a << "\n";
	std::cout << "矩阵 b：\n" << b << "\n";
	std::cout << "矩阵相加：\n" << (a + b) << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 矩阵乘法; Multiplying Matrices and Vectors\n";
	std::cout << "// --------------------------------------------------\n\n";
	auto m1 = torch::tensor({{1.0,3.0},{1.0,0.0}});
	auto m2 = torch::tensor({{1.0,2.0},{5.0,0.0}});
	std::cout << "按矩阵乘法规则 standard product of two matrices - matmul：\n" << torch::matmul(m1, m2) << '\n';
	std::cout << "按矩阵乘法规则 standard product of two matrices - mm：\n" << torch::mm(m1, m2) << '\n';
	std::cout << "按逐元素相乘 element-wise product or Hadamard product：\n"   << torch::mul(m1, m2) << '\n';
	std::cout << "按逐元素相乘 element-wise product or Hadamard product：\n"   << m1*m2 << '\n';
	auto v1 = torch::tensor({1.0,2.0});
	auto v2 = torch::tensor({4.0,5.0});
	std::cout << "向量内积：Vector Dot Product\n" << torch::dot(v1, v2) << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 单位矩阵; Identity Matrix\n";
	std::cout << "// --------------------------------------------------\n\n";
	auto I = torch::eye(3);
	std::cout << "单位矩阵; Identity Matrix\n" << I << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 矩阵的逆; Inversion\n";
	std::cout << "// --------------------------------------------------\n\n";
	A = torch::tensor({{1.0,2.0},{3.0,4.0}});
	auto A_inv = torch::inverse(A);
	std::cout << "A 的逆矩阵:\n" << A_inv << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 范数; norm\n";
	std::cout << "// --------------------------------------------------\n\n";
	a = torch::tensor({1.0, 3.0});
	std::cout << "向量 2 范数:\n" << torch::norm(a, 2) << "\n";
	std::cout << "向量 1 范数:\n" << torch::norm(a, 1) << "\n";
	std::cout << "向量无穷范数:\n" << torch::norm(a, INFINITY) << "\n";

	a = torch::tensor({{1.0,3.0}, {2.0,1.0}});
	auto F = torch::sqrt(torch::dot(a.reshape({-1}), a.reshape({-1})));
	std::cout << "矩阵 F 范数:\n" << F << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 特征值分解; Eigendecomposition\n";
	std::cout << "// --------------------------------------------------\n\n";
	A = torch::tensor({
		{1.0,2.0,3.0},
		{4.0,5.0,6.0},
		{7.0,8.0,9.0}});
	//# 计算特征值
	std::cout << "特征值:\n" << torch::linalg::eigvals(A) << '\n';
	//# 计算特征值和特征向量
	torch::Tensor eigvals, eigvectors;
	std::tie(eigvals, eigvectors)= torch::linalg::eig(A);
	std::cout << "特征值:\n" << eigvals << '\n';
	std::cout << "特征向量:\n" << eigvectors << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 奇异值分解; Singular Value Decomposition\n";
	std::cout << "// --------------------------------------------------\n\n";
	A = torch::tensor({{1.0,2.0,3.0}, {4.0,5.0,6.0}});
	torch::Tensor U, D, V;
	std::tie(U,D,V) = torch::svd(A);
	std::cout << "U:\n" << U << '\n';
	std::cout << "D:\n" << D << '\n';
	std::cout << "V:\n" << V << "\n\n";

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 主成分分析; Principal Components Analysis\n";
	std::cout << "// --------------------------------------------------\n\n";
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

    int64_t n_samples = X.size(0);
     X = X.to(torch::kDouble);
     c10::OptionalArrayRef<long int> dm = {0};
    torch::Tensor centering_X = X - X.mean(dm);
    std::cout << "centering_X:\n" << centering_X.sizes() << '\n';
    std::cout << "X.mean(dm):\n" << X.mean(dm) << '\n';
    torch::Tensor covariance_matrix = torch::mm(centering_X.t(), centering_X)/(n_samples-1);
    std::cout << "covariance_matrix:\n" << covariance_matrix << '\n';

    // 对协方差矩阵进行特征值分解
    std::tie(eigvals, eigvectors) = torch::linalg::eig(covariance_matrix);
    // 对特征值（特征向量）从大到小排序
    eigvals = eigvals.to(torch::kDouble);
    eigvectors = eigvectors.to(torch::kDouble);
    auto idx = eigvals.argsort(-1, true);
    std::cout << "idx:\n" << idx << '\n';
    std::cout << "eigvals:\n" << eigvals << '\n';
    //idx = eigenvalues.argsort()[::-1]
    int64_t n_components = 2;
    auto eigenvalues = torch::index_select(eigvals, 0, idx ).index({Slice(0, n_components)});
    std::cout << "eigenvalues:\n" << eigenvalues << '\n';
    auto eigenvectors = torch::index_select(eigvectors, 1, idx ).index({Slice(), Slice(0, n_components)});
    std::cout << "eigenvectors:\n" << eigenvectors.sizes()<< " " << eigenvectors.dtype() << '\n';
    // 得到低维表示
    auto X_transformed = X.mm(eigenvectors);
    auto X_trs = torch::mm(eigenvectors.t(), centering_X.t()).t();

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
    h->size(1200, 600);
    h->add_axes(false);
    h->reactive_mode(false);
    h->tiledlayout(1, 2);
    h->position(0, 0);

    double sz = 12;
    std::vector<double> xx, yy;
    std::vector<int> c;
    for(int64_t i = 0; i < y.size(0); i++) {
    	xx.push_back(X_transformed.index({i, 0}).data().item<double>());
    	yy.push_back(X_transformed.index({i, 1}).data().item<double>());
    	c.push_back(y[i].data().item<int>());
    }
    auto ax1 = h->nexttile();
    auto st = scatter(ax1, xx, yy, sz, c);
    st->marker_face(true);
    ax1->xlabel("PC 1");
    ax1->ylabel("PC 2");
    ax1->title("origin data");
    ax1->grid(on);
    c10::OptionalArrayRef<long int> dim = {0};
    torch::Tensor xy_0 = torch::mean(torch::index_select(X_transformed, 0, index_0), dim);
    torch::Tensor xy_1 = torch::mean(torch::index_select(X_transformed, 0, index_1), dim);
    torch::Tensor xy_2 = torch::mean(torch::index_select(X_transformed, 0, index_2), dim);
    ax1->text(xy_0[0].data().item<double>(), xy_0[1].data().item<double>(), "← setosa")->font_size(12);
    ax1->text(xy_1[0].data().item<double>(), xy_1[1].data().item<double>(), "← versicolor")->font_size(12);
    ax1->text(xy_2[0].data().item<double>(), xy_2[1].data().item<double>(), "← virginica")->font_size(12);

    auto ax2 = h->nexttile();
    std::vector<double> xx2, yy2;
    std::vector<int> c2;
    for(int64_t i = 0; i < y.size(0); i++) {
    	xx2.push_back(X_trs.index({i, 0}).data().item<double>());
    	yy2.push_back(X_trs.index({i, 1}).data().item<double>());
    	c2.push_back(y[i].data().item<int>());
    }
    auto st2 = scatter(ax2, xx2, yy2, sz, c2);
    st2->marker_face(true);
    ax2->xlabel("PC 1");
    ax2->ylabel("PC 2");
    ax2->title("centered data");
    ax2->grid(on);
    c10::OptionalArrayRef<long int> dim2 = {0};
    xy_0 = torch::mean(torch::index_select(X_trs, 0, index_0), dim2);
    xy_1 = torch::mean(torch::index_select(X_trs, 0, index_1), dim2);
    xy_2 = torch::mean(torch::index_select(X_trs, 0, index_2), dim2);
    ax2->text(xy_0[0].data().item<double>(), xy_0[1].data().item<double>(), "← setosa")->font_size(12);
    ax2->text(xy_1[0].data().item<double>(), xy_1[1].data().item<double>(), "← versicolor")->font_size(12);
    ax2->text(xy_2[0].data().item<double>(), xy_2[1].data().item<double>(), "← virginica")->font_size(12);

    show();

	std::cout << "Done\n";
	file.close();
	return 0;
}


