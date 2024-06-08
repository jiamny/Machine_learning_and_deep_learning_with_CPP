/*
 * CRF.cpp
 *
 *  Created on: Jun 3, 2024
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
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

bool cmp(std::pair<torch::Tensor, double> a, std::pair<torch::Tensor, double> b) {
	return a.second > b.second;
}


class CRFMatrix {
public:
	std::vector<torch::Tensor> M;
	int start =0, stop = 0;
	std::vector<std::pair<torch::Tensor, double>> path_prob;

	CRFMatrix(std::vector<torch::Tensor> _M, int _start, int _stop) {
        // 随机矩阵
        M = _M;
        start = _start;
        stop = _stop;
        path_prob.clear();
	}

	std::vector<torch::Tensor> _create_path(void) {
        // 按照图11.6的状态路径图，生成路径
        // 初始化start结点
		std::vector<torch::Tensor> path;
		int start = 2, stop = 2;
	    path.push_back(torch::tensor({start}, torch::kInt32));
	    //std::cout << path[0] << '\n';

	    for(int i = 1; i < M.size(); i++) {
	    	std::vector<torch::Tensor> paths;
	        for(auto& r : path) {
	            auto temp = r.t();
	            // 添加状态结点1
	            paths.push_back(torch::cat({temp, torch::tensor({1}, torch::kInt32)}, 0));
	            // 添加状态结点2
	            paths.push_back(torch::cat({temp, torch::tensor({2}, torch::kInt32)}, 0));
	        }
	        path = paths;
	    }

	    // 添加stop结点
	    //path = [np.append(r, self.stop) for _, r in enumerate(path)]
	    std::vector<torch::Tensor> paths;
	    for(auto& r : path) {
	    	auto t = torch::cat({r, torch::tensor({stop}, torch::kInt32)}, 0);
	    	//std::cout << "t:\n" << t << '\n';
	    	paths.push_back(t.clone());
	    }
        return paths;
    }

    void fit(void) {
    	std::vector<torch::Tensor> path = _create_path();
    	std::vector<std::pair<torch::Tensor, double>> pr;
        for(auto& row : path) {
            double p = 1.;
            for(auto& i : range((static_cast<int>(row.size(0)) - 1), 0)) {
                int a = row[i].data().item<int>();
                int b = row[i + 1].data().item<int>();
                // 根据公式11.24，计算条件概率
                torch::Tensor m = M[i];
                p *= (m[a - 1][b - 1]).data().item<double>();
            }
            std::pair<torch::Tensor, double> t = std::make_pair(row, p);
            pr.push_back(t);
        }
        // 按照概率从大到小排列
        std::sort(pr.begin(), pr.end(), cmp);

        path_prob = pr;
    }

    void print(void) {
        // 打印结果
        printf("以start=%d为起点stop=%d为终点的所有路径的状态序列y的概率为：\n",  start, stop);
        for(auto& t : path_prob ) {
        	torch::Tensor path = t.first;
        	double p  = t.second;
        	//join([str(x) for x in path])
        	std:: string str = "    路径为：" + std::to_string(path[0].data().item<int>());
        	for(int i = 1; i < path.size(0); i++)
        		str += ("->" + std::to_string(path[i].data().item<int>()));

            printf("%s%s ", str.c_str(), " ");
            printf("概率为：%f\n", p);
        }
        torch::Tensor path = path_prob[0].first;
        double p  = path_prob[0].second;
        std:: string str = std::to_string(path[0].data().item<int>());
        for(int i = 1; i < path.size(0); i++)
        	str += ("->" + std::to_string(path[i].data().item<int>()));
        printf("概率最大[%f]的状态序列为: %s\n", p, str.c_str());
    }

};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// 创建随机矩阵
	torch::Tensor M1 = torch::tensor({{0., 0.}, {0.5, 0.5}}, torch::kDouble);
	torch::Tensor M2 = torch::tensor({{0.3, 0.7}, {0.7, 0.3}}, torch::kDouble);
	torch::Tensor M3 = torch::tensor({{0.5, 0.5}, {0.6, 0.4}}, torch::kDouble);
	torch::Tensor M4 = torch::tensor({{0., 1.}, {0., 1.}}, torch::kDouble);
	std::vector<torch::Tensor> M = {M1, M2, M3, M4};

	// 构建条件随机场的矩阵模型
	auto crf = CRFMatrix(M, 2, 2);
	// 得到所有路径的状态序列的概率
	crf.fit();
	// 打印结果
	crf.print();

	std::cout << "Done!\n";
	return 0;
}





