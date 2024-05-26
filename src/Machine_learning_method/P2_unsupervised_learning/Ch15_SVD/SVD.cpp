/*
 * SVD.cpp
 *
 *  Created on: May 3, 2024
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
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

//基于矩阵分解的结果，复原矩阵
torch::Tensor rebuildMatrix(torch::Tensor U, torch::Tensor sigma, torch::Tensor V, double epsilon=1e-10) {

	if(sigma.size(0) < U.size(1)) {
		std::vector<double> d = tensorTovector(sigma);
		for(int i = 0; i < (U.size(1) - sigma.size(0)); i++) {
			d.push_back(0.);
		}
		sigma = torch::from_blob(d.data(), {int(d.size())}, at::TensorOptions(torch::kDouble)).clone();
	}

	torch::Tensor D = torch::diag(sigma);
	torch::Tensor a = torch::mm(U, D);
	a = a.masked_fill( torch::abs(a) < epsilon, 0.);
    a = torch::mm(a, V.t());
    return a;
}

//基于特征值的大小，对特征值以及特征向量进行排序。倒序排列
std::pair<torch::Tensor, torch::Tensor> sortByEigenValue(torch::Tensor Eigenvalues, torch::Tensor EigenVectors) {

	auto index = Eigenvalues.argsort(-1, true);
	torch::Tensor eigenvalues = torch::index_select(Eigenvalues, 0, index );
	torch::Tensor eigenvectors = torch::index_select(EigenVectors, 1, index );
    return std::make_pair(eigenvalues, eigenvectors);
}

//对一个矩阵进行奇异值分解
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SVD(torch::Tensor A) {
	torch::Tensor AT = A.t().clone();
	torch::Tensor AT_A = torch::mm(AT, A);
	torch::Tensor A_AT = torch::mm(A, AT);

    //然后求右奇异向量
    torch::Tensor eigv_AT_A, eigvt_AT_A;
    std::tie(eigv_AT_A, eigvt_AT_A)= torch::linalg::eig(AT_A);

    torch::Tensor eigv_A_AT, eigvt_A_AT;
    std::tie(eigv_A_AT, eigvt_A_AT)= torch::linalg::eig(A_AT);

    eigv_AT_A = eigv_AT_A.to(torch::kDouble);
    eigvt_AT_A = eigvt_AT_A.to(torch::kDouble);
    eigv_A_AT = eigv_A_AT.to(torch::kDouble);
    eigvt_A_AT = eigvt_A_AT.to(torch::kDouble);

    std::tie(eigv_AT_A, eigvt_AT_A) =  sortByEigenValue(eigv_AT_A, eigvt_AT_A);

	//求奇异值
    torch::Tensor D = eigv_AT_A.clone();
    for(int64_t i = 0; i < D.size(0); i++) {
    	double t = D[i].data().item<double>();
    	c10::ArrayRef<at::indexing::TensorIndex> ii = {i};
    	if( t < 0 ) {
    		D.index_put_(ii, -0.);
    	} else {
    		double x = std::sqrt(t);
    		D.index_put_(ii, x);
    	}
    }

    // U =  eigenvectors of AAT
    torch::Tensor U = eigvt_A_AT.clone();

    // V = eigenvectors of AT A
    torch::Tensor V = eigvt_AT_A.clone();

    return std::make_tuple(U, D, V);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Tensor A = torch::tensor({{1, 1, 1, 2, 2}, {0, 0, 0, 3, 3},
											  {0, 0, 0, 1, 1}, {1, 1, 1, 0, 0},
											  {2, 2, 2, 0, 0}, {5, 5, 5, 0, 0},
											  {1, 1, 1, 0, 0}}, torch::kDouble);

	torch::Tensor U, D, V;
	std::tie(U, D, V) = svd(A);
	std::cout << "U:\n" << U << '\n';
	std::cout << "D:\n" << D << '\n';
	std::cout << "V:\n" << V << "\n\n";

	torch::Tensor RA = rebuildMatrix(U, D, V);
	std::cout << "rebuiled matrix:\n" << RA << "\n\n";

	torch::Tensor X = torch::tensor({{1, 1},
                  {2, 2},
                  {0, 0}}, torch::kDouble);

	torch::Tensor _U, _D, _V;
	std::tie(_U,_D,_V) = torch::svd(X);
	std::cout << "_U:\n" << _U << '\n';
	std::cout << "_D:\n" << _D << '\n';
	std::cout << "_V:\n" << _V << "\n\n";

	torch::Tensor RX = rebuildMatrix(_U, _D, _V);
	std::cout << "rebuiled matrix:\n" << RX << "\n\n";

	/*
	torch::Tensor _U, _D, _V;
	std::tie(_U,_D,_V) = torch::svd(A);
	std::cout << "U:\n" << _U << '\n';
	std::cout << "D:\n" << _D << '\n';
	std::cout << "V:\n" << _V << "\n\n";
	 */
	std::cout << "Done!\n";
	return 0;
}


