/*
 * LatentSemanticAnalysis.cpp
 *
 *  Created on: May 19, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;


std::pair<torch::Tensor, torch::Tensor> em_for_plsa(torch::Tensor X, int K, int max_iter=100, int random_state=123) {
    //概率潜在语义模型参数估计的EM算法
    //:param X: 单词-文本共现矩阵
    //:param K: 话题数量
    //:param max_iter: 最大迭代次数
    //:param random_state: 随机种子
    //:return: P(w_i|z_k)和P(z_k|d_j)

    int n_features = X.size(0);
    int n_samples = X.size(1);

    //计算n(d_j)
    std::vector<torch::Tensor> N;
    for(auto& j : range(n_samples, 0)) {
    	N.push_back(torch::sum(X.index({Slice(), j})));
    }

    // 设置参数P(w_i|z_k)和P(z_k|d_j)的初始值
	torch::manual_seed(random_state);
	torch::Tensor P1 = torch::rand({n_features, K}, torch::kDouble);  // P(w_i|z_k)
	torch::Tensor P2 = torch::rand({K, n_samples}, torch::kDouble);   // P(z_k|d_j)
	/*
	torch::Tensor P1 = torch::tensor( {{0.55, 0.72, 0.6 },
						  {0.54, 0.42, 0.65},
						  {0.44, 0.89, 0.96},
						  {0.38, 0.79, 0.53},
						  {0.57, 0.93, 0.07},
						  {0.09, 0.02, 0.83},
						  {0.78, 0.87, 0.98},
						  {0.8,  0.46, 0.78},
						  {0.12, 0.64, 0.14},
						  {0.94, 0.52, 0.41},
						  {0.26, 0.77, 0.46}}, torch::kDouble);
    torch::Tensor P2 = torch::tensor(
    		{{0.57, 0.02, 0.62, 0.61, 0.62, 0.94, 0.68, 0.36, 0.44},
    		 {0.7,  0.06, 0.67, 0.67, 0.21, 0.13, 0.32, 0.36, 0.57},
    		 {0.44, 0.99, 0.1,  0.21, 0.16, 0.65, 0.25, 0.47, 0.24}}, torch::kDouble);
	*/
    for(auto & _ : range(max_iter, 0) ) {
        // E步
    	torch::Tensor P = torch::zeros({n_features, n_samples, K}, torch::kDouble);
        for( auto& i : range(n_features, 0) ) {
            for(auto& j : range(n_samples, 0) ) {
                for(auto& k : range(K, 0) ) {
                    P[i][j][k] = P1[i][k] * P2[k][j];
                }
                P[i][j] /= torch::sum(P[i][j]);
            }
        }
        // M步
        for(auto& k : range(K, 0)) {
            for(auto& i : range(n_features, 0) ) {
            	double s = 0.;
                for(auto& j : range(n_samples, 0)) {
                	 s += (X[i][j] * P[i][j][k]).data().item<double>();
                }
                P1[i][k] = s;
            }
            double t = torch::sum(P1.index({Slice(), k})).data().item<double>();
            //P1[:, k] /= t;
            for(auto& n : range(n_features, 0))
            	P1[n][k] /= t;
        }

        for(auto& k : range(K, 0) ) {
            for(auto& j : range(n_samples, 0) ) {
            	double s = 0.;
            	for(auto& i : range(n_features, 0))
            		s += (X[i][j] * P[i][j][k]).data().item<double>();
            	s /= N[j].data().item<double>();
            	P2[k][j] = s;
                //P2[k][j] = torch::sum([X[i][j] * P[i][j][k] for i in range(n_features)]) / N[j]
            }
        }
    }

    return std::make_pair(P1, P2);
}

std::pair<torch::Tensor, torch::Tensor> nmp_training(torch::Tensor X, int k,
		int max_iter=100, double tol=1e-4, int random_state=1) {
    //非负矩阵分解的迭代算法（平方损失）
    //:param X: 单词-文本矩阵
    //:param k: 文本集合的话题个数k
    //:param max_iter: 最大迭代次数
    //:param tol: 容差
    //:param random_state: 随机种子
    //:return: 话题矩阵W,文本表示矩阵H

    int n_features = X.size(0);
    int n_samples = X.size(1);

    // 初始化
    torch::manual_seed(random_state);
    torch::Tensor W = torch::rand({n_features, k}, torch::kDouble);
    torch::Tensor H = torch::rand({k, n_samples}, torch::kDouble);
    /*
    torch::Tensor W = torch::tensor(
    		{{0.55, 0.72, 0.6 },
    	 	 {0.54, 0.42, 0.65},
			 {0.44, 0.89, 0.96},
			 {0.38 ,0.79, 0.53},
			 {0.57, 0.93, 0.07},
			 {0.09, 0.02, 0.83},
			 {0.78, 0.87, 0.98},
			 {0.8,  0.46, 0.78},
			 {0.12, 0.64, 0.14},
			 {0.94, 0.52, 0.41},
			 {0.26, 0.77, 0.46}}, torch::kDouble);
    torch::Tensor H = torch::tensor(
    		{{0.57, 0.02, 0.62, 0.61, 0.62, 0.94, 0.68, 0.36, 0.44},
    		 {0.7,  0.06, 0.67, 0.67, 0.21, 0.13, 0.32, 0.36, 0.57},
    		 {0.44, 0.99, 0.1,  0.21, 0.16, 0.65, 0.25, 0.47, 0.24}}, torch::kDouble);
	*/

    // 计算当前平方损失
    double last_score = torch::sum(torch::square(X - torch::mm(W, H))).data().item<double>();

    // 迭代
    for(auto& _ : range(max_iter, 0)) {

        // 更新W的元素
    	torch::Tensor A = torch::mm(X, H.t());  				// X H^T
    	torch::Tensor B = torch::mm(torch::mm(W, H), H.t());	// W H H^T
        for(auto& i : range(n_features, 0) ) {
            for(auto& l : range(k, 0)) {
                W[i][l] *= A[i][l] / B[i][l];
            }
        }

        // 更新H的元素
        torch::Tensor C = torch::mm(W.t(), X);  				// W^T X
        torch::Tensor D = torch::mm(torch::mm(W.t(), W), H);	// W^T W H
        for(auto& l : range(k, 0)) {
            for(auto& j : range(n_samples, 0)) {
                H[l][j] *= C[l][j] / D[l][j];
            }
        }

        // 检查迭代更新量是否已小于容差
        double now_score = torch::sum(torch::square(X - torch::mm(W, H))).data().item<double>();
        std::cout << "(last_score - now_score): " << (last_score - now_score) << '\n';
        if((last_score - now_score) < tol)
            break;

        last_score = now_score;
    }

    return std::make_pair(W, H);
}

std::pair<torch::Tensor, torch::Tensor> lsa_by_svd(torch::Tensor X, int k) {
    //利用矩阵奇异值分解的潜在语义分析

    //:param X: 单词文本矩阵
    //:param x: 目标话题数量
    //:return: 话题向量空间, 文本集合在话题向量空间的表示
	torch::Tensor U, S, V;
    std::tie(U, S, V) = torch::svd(X);  // 奇异值分解
    torch::Tensor _U = U.index({Slice(), Slice(0, k)}).clone();
    torch::Tensor _S = torch::diag(S.index({Slice(0, k)})).clone();
    std::cout << "_S:\n" << _S << '\n';
    torch::Tensor _V = V.index({Slice(0, k), Slice()}).clone();

    return std::pair(_U, torch::mm(_S, _V));
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  非负矩阵分解的迭代算法（平方损失）\n";
	std::cout << "// --------------------------------------------------\n";

    torch::Tensor X = torch::tensor({{0, 0, 1, 1, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 1, 0, 0, 1},
                  {0, 1, 0, 0, 0, 0, 0, 1, 0},
                  {0, 0, 0, 0, 0, 0, 1, 0, 1},
                  {1, 0, 0, 0, 0, 1, 0, 0, 0},
                  {1, 1, 1, 1, 1, 1, 1, 1, 1},
                  {1, 0, 1, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 1, 0, 1},
                  {0, 0, 0, 0, 0, 2, 0, 0, 1},
                  {1, 0, 1, 0, 0, 0, 0, 1, 0},
                  {0, 0, 0, 1, 1, 0, 0, 0, 0}}, torch::kDouble);
    torch::Tensor W, H;
    std::tie(W, H) = nmp_training(X, 3);
    std::cout << "W:\n" << W << '\n';
    std::cout << "H:\n" << H << '\n';

    torch::Tensor Y = torch::mm(W, H);
    std::cout << "Y:\n" << Y << '\n';
    std::cout << "sum(square(X - Y)): " << torch::sum(torch::square(X - Y)) << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// 利用矩阵奇异值分解的潜在语义分析\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor U, SV;
    std::tie(U, SV) = lsa_by_svd(X, 3);

    std::cout << "U:\n" << U << '\n';
    std::cout << "SV:\n" << SV << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  概率潜在语义模型参数估计的EM\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor R1, R2;
	std::tie(R1, R2) = em_for_plsa(X, 3);
	std::cout << "R1:\n" << R1 << '\n';
	std::cout << "R2:\n" << R2.masked_fill_(R2 < 1e-10, 0.0) << '\n';

	std::cout << "Done!\n";
	return 0;
}

