/*
 * PageRank.cpp
 *
 *  Created on: Jun 12, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <regex>
#include <iostream>
#include <iomanip>
#include <set>
#include <unistd.h>
#include <random>
#include <ctime>

#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;


torch::Tensor page_rank_basic(torch::Tensor M, torch::Tensor R0, double tol=1e-8, int max_iter=1000) {
    /*
    迭代求解基本定义的PageRank
    :param M: 转移矩阵
    :param R0: 初始分布向量
    :param max_iter: 最大迭代次数
    :return: Rt: 极限向量
    */

    for(auto&  _ : range(max_iter, 0) ) {
    	torch::Tensor R1 = torch::mm(M, R0);

    	// 判断迭代更新量是否小于容差
    	if(torch::sum(torch::abs(R0 - R1)).data().item<double>() < tol)
    	   break;

    	 R0 = R1.clone();
    }
    return R0;
}

// PageRank的迭代算法
torch::Tensor page_rank_iteration_method(torch::Tensor M, torch::Tensor R0, int n, double d, double eps) {
	int t = 0;  					//用来累计迭代次数
	torch::Tensor R = R0.clone();	//对R向量进行初始化
	bool judge = false;  			// 用来判断是否继续迭代
	while( ! judge ) {
		torch::Tensor next_R = d * torch::matmul(M, R) + (1 - d) / n * torch::ones({n, 1}, torch::kDouble);  //计算新的R向量
		torch::Tensor diff = torch::norm(R - next_R);  // 计算新的R向量与之前的R向量之间的距离，这里采用的是欧氏距离
		if( diff.data().item<double>() < eps)  		//若两向量之间的距离足够小
			judge = true;		// 则停止迭代
		R = next_R.clone();  	// 更新R向量
		t += 1;					//迭代次数加一
		if( t % 10 == 0)
			std::cout << "迭代次数：" << t << '\n';
	}

	R = R / torch::sum(R); 		// 对R向量进行规范化，保证其总和为1，表示各节点的概率分布
	return R;
}

// PageRank的幂法
torch::Tensor page_rank_power_method(torch::Tensor M, torch::Tensor R0, int n, double d, double eps) {
	int t = 0;  	 //用来累计迭代次数
	torch::Tensor x = R0.clone();   // 对x向量进行初始化
	bool judge = false;  			// 用来判断是否继续迭代
	torch::Tensor A = d * M + (1 - d) / n * torch::eye(n).to(torch::kDouble);	// 计算A矩阵，其中np.eye(n)用来创建n阶单位阵E
	while(! judge) {
		torch::Tensor next_y = torch::matmul(A, x); 		// 计算新的y向量
		torch::Tensor next_x = next_y / torch::norm(next_y); // 对新的y向量规范化得到新的x向量
		torch::Tensor diff = torch::norm(x - next_x);		//计算新的x向量与之前的x向量之间的距离，这里采用的是欧氏距离
	    if( diff.data().item<double>() < eps) {	// 若两向量之间的距离足够小
	        judge = true;	// 则停止迭代
	    }
	    x = next_x;  		// 更新x向量
	    t += 1;				// 迭代次数加一
	    if( t % 10 == 0)
	    	std::cout << "迭代次数：" << t << '\n';
	}
	x = x / torch::sum(x);  // 对R向量进行规范化，保证其总和为1，表示各节点的概率分布
	return x;
}

torch::Tensor page_rank_algebra_method(torch::Tensor M, double d=0.8) {
    /*PageRank的代数算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    */
    int n_components = M.size(0);

    // 计算第一项：(I-dM)^-1
    torch::Tensor r1 = torch::linalg_inv(torch::eye(n_components).to(torch::kDouble) - d * M);

    // 计算第二项：(1-d)/n 1
    torch::Tensor r2 = torch::ones({n_components, 1}, torch::kDouble)*((1 - d) / n_components);

    return torch::mm(r1, r2);
}

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
    torch::manual_seed(123);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Page rank basic \n";
	std::cout << "// --------------------------------------------------\n";

	// 使用例21.1的转移矩阵M
	torch::Tensor M = torch::tensor(
			 {{0., 1. / 2, 1., 0.},
              {1. / 3, 0., 0., 1. / 2},
              {1. / 3, 0., 0., 1. / 2},
              {1. / 3, 1. / 2, 0., 0.}}, torch::kDouble);

	// 使用5个不同的初始分布向量R0
	for(auto& _ : range(5, 0)) {
		torch::Tensor R0 = torch::rand({4, 1}, torch::kDouble);
		R0 = R0 / torch::norm(R0, 1);
		torch::Tensor Rt = page_rank_basic(M, R0);
		std::cout << "R0 =\n" << R0 << '\n';
		std::cout << "Rt =\n" << Rt << '\n';
	}

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Page rank iteration method \n";
	std::cout << "// --------------------------------------------------\n";
	int n = 7;  // 有向图中一共有7个节点
	double d = 0.85;  // 阻尼因子根据经验值确定，这里我们随意给一个值
	M = torch::tensor({{0., 1./4, 1./3, 0., 0., 1./2, 0.},
	              {1./4, 0., 0., 1./5, 0., 0., 0.},
	              {0., 1./4, 0., 1./5, 1./4, 0., 0.},
	              {0., 0., 1./3, 0., 1./4, 0., 0.},
	              {1./4, 0., 0., 1./5, 0., 0., 0.},
	              {1./4, 1./4, 0., 1./5, 1./4, 0., 0.},
	              {1./4, 1./4, 1./3, 1./5, 1./4, 1./2, 0.}}, torch::kDouble);	// 根据有向图中各节点的连接情况写出转移矩阵

	torch::Tensor R0 = torch::full({7, 1}, 1./7).to(torch::kDouble);  //设置初始向量R0，R0是一个7*1的列向量，因为有7个节点，我们把R0的每一个值都设为1/7
	double eps = 0.000001;  //设置计算精度
	torch::Tensor R = page_rank_iteration_method(M, R0, n, d, eps);
	std::cout << "PageRank iteration method: \n";
	printVector(tensorTovector(R));

	torch::Tensor P = torch::tensor({{0., 0., 1.},
                  {1. / 2, 0., 0.},
                  {1. / 2, 1., 0.}}, torch::kDouble);
	torch::Tensor P0 = torch::full({3, 1}, 1./3).to(torch::kDouble);

	R = page_rank_iteration_method(P, P0, 3, d, eps);
	std::cout << "PageRank iteration method: \n";
	printVector(tensorTovector(R));

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Page rank power method \n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor x = page_rank_power_method(M, R0, n, d, eps);
	std::cout << "PageRank power method: \n";
	printVector(tensorTovector(x));

	std::cout << "P:\n" << P << '\n';
	x = page_rank_power_method(P, P0, 3, d, eps);
	std::cout << "PageRank power method: \n";
	printVector(tensorTovector(x));

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Page rank algebra method \n";
	std::cout << "// --------------------------------------------------\n";
	x = page_rank_algebra_method(M, 0.8);
	std::cout << "PageRank algebra method: \n";
	printVector(tensorTovector(x));

	x = page_rank_algebra_method(P, 0.8);
	std::cout << "PageRank algebra method: \n";
	printVector(tensorTovector(x));

	std::cout << "Done!\n";
	return 0;
}
