/*
 * Zero_sum_game.cpp
 *
 *  Created on: Jul 20, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

int  minmax_function(torch::Tensor A) {
    /*
    从收益矩阵中计算minmax的算法
    :param A: 收益矩阵
    :return: 计算得到的minmax结果
    */
    std::vector<int> index_max;
    for(auto& i : range(static_cast<int>(A.size(0)), 0)) {
        // 计算每一行的最大值
        index_max.push_back(A.index({i, Slice()}).max().data().item<int>());
    }

    // 计算每一行的最大值中的最小值
    auto minmax = std::minmax_element(index_max.begin(), index_max.end());
    return minmax.first[0];
}


int maxmin_function(torch::Tensor  A) {
    /*
    从收益矩阵中计算maxmin的算法
    :param A: 收益矩阵
    :return: 计算得到的maxmin结果
    */
	std::vector<int> column_min;
	for(auto& i : range(static_cast<int>(A.size(0)), 0)) {
        // 计算每一列的最小值
        column_min.push_back(A.index({Slice(), i}).min().data().item<int>());
	}

    // 计算每一列的最小值中的最大值
    auto minmax = std::minmax_element(column_min.begin(), column_min.end());
    return minmax.second[0];
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// 创建收益矩阵
	torch::Tensor A = torch::tensor({{-1, 2}, {4, 1}}, torch::kInt32);
	// 计算maxmin
	int maxmin = maxmin_function(A);

	// 计算minmax
	int minmax = minmax_function(A);
	// 输出结果
	printf("maxmin = %d\n", maxmin);
	printf("minmax = %d\n", minmax);

	std::cout << "Done!\n";
	return 0;
}


