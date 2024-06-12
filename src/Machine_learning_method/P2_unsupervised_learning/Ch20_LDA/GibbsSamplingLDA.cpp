/*
 * GibbsSamplingLDA.cpp
 *
 *  Created on: Jun 9, 2024
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
#include <limits>
#include <ctime>
#include <cstdlib>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class GibbsSamplingLDA {
public:
	GibbsSamplingLDA(int _iter_max=1000) {
        iter_max = _iter_max;
        weights_.clear();
	}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fit(torch::Tensor words, int K) {
        /*
        :param words: 单词-文本矩阵
        :param K: 话题个数
        :return: 文本话题序列z
        */
        // M, Nm分别为文本个数和单词个数
        words = words.t();
        int M = words.size(0), Nm = words.size(1);

        // 初始化超参数alpha, beta，其中alpha为文本的话题分布相关参数，beta为话题的单词分布相关参数
        torch::Tensor alpha = torch::ones({K}, torch::kDouble) * (1.0 / K);
		torch::Tensor beta = torch::ones({Nm}, torch::kDouble) * (1.0 / Nm);

        // 初始化参数theta, varphi，其中theta为文本关于话题的多项分布参数，varphi为话题关于单词的多项分布参数
		torch::Tensor theta = torch::zeros({M, K}, torch::kDouble);
		torch::Tensor varphi = torch::zeros({K, Nm}, torch::kDouble);

        // 输出文本的话题序列z
		torch::Tensor z = torch::zeros(words.sizes(), torch::kInt32);

        // (1)设所有计数矩阵的元素n_mk、n_kv，计数向量的元素n_m、n_k初值为 0
		torch::Tensor n_mk = torch::zeros({M, K}, torch::kDouble);
		torch::Tensor n_kv = torch::zeros({K, Nm}, torch::kDouble);
		torch::Tensor n_m = torch::zeros({M}, torch::kDouble);
		torch::Tensor n_k = torch::zeros({K}, torch::kDouble);

        // (2)对所有M个文本中的所有单词进行循环
        for(auto& m : range(M, 0)) {
            for(auto& v : range(Nm, 0)) {
                // 如果单词v存在于文本m
                if( words[m][v].data().item<int>() != 0 ) {
                    // (2.a)抽样话题
                	std::vector<int> d = random_choice(1, range(K, 0));
                    z[m][v] = d[0];	//np.random.choice(list(range(K)))
                    // 增加文本-话题计数
                    n_mk[m][(z[m][v]).data().item<int>()] += 1;
                    // 增加文本-话题和计数
                    n_m[m] += 1;
                    // 增加话题-单词计数
                    n_kv[(z[m][v]).data().item<int>()][v] += 1;
                    // 增加话题-单词和计数
                    n_k[(z[m][v]).data().item<int>()] += 1;
                }
            }
        }
        // (3)对所有M个文本中的所有单词进行循环，直到进入燃烧期
        double zi = 0;
        for(auto& i : range(iter_max, 0)) {
            for(auto& m : range(M, 0)) {
                for(auto& v : range(Nm, 0)) {
                    // (3.a)如果单词v存在于文本m，那么当前单词是第v个单词，话题指派z_mv是第k个话题
                    if( words[m][v].data().item<int>() != 0) {
                        // 减少计数
                        n_mk[m][(z[m][v]).data().item<int>()] -= 1;
                        n_m[m] -= 1;
                        n_kv[(z[m][v]).data().item<int>()][v] -= 1;
                        n_k[(z[m][v]).data().item<int>()] -= 1;

                        // (3.b)按照满条件分布进行抽样
                        double max_zi_value = std::numeric_limits<double>::infinity();
                        double max_zi_index = z[m][v].data().item<int>();
                        for(auto& k : range(K, 0)) {
                        	torch::Tensor t1 = ((n_kv[k][v] + beta[v]) / (n_kv.index({Slice(k, None)}).sum() + beta.sum()));
                        	torch::Tensor t2 = ((n_mk[m][k] + alpha[k]) / (n_mk.index({Slice(m, None)}).sum() + alpha.sum()));
                        	zi = (t1  * t2).data().item<double>();

							// 得到新的第 k‘个话题，分配给 z_mv
							if( max_zi_value < zi ) {
								max_zi_value = zi;
								max_zi_index = k;
								z[m][v] = max_zi_index;
							}
                        }
                        // (3.c) (3.d)增加计数并得到两个更新的计数矩阵的n_kv和n_mk
                        n_mk[m][(z[m][v]).data().item<int>()] += 1;
                        n_m[m] += 1;
                        n_kv[(z[m][v]).data().item<int>()][v] += 1;
                        n_k[(z[m][v]).data().item<int>()] += 1;
                    }
                }
            }
        }
        // (4)利用得到的样本计数，计算模型参数
        for(auto& m : range(M, 0)) {
            for(auto& k : range(K, 0)) {
                theta[m][k] = (n_mk[m][k] + alpha[k]) / (n_mk.index({Slice(m, None)}).sum() + alpha.sum());
            }
        }

        for(auto& k : range(K, 0)) {
            for(auto& v : range(Nm, 0)) {
                varphi[k][v] = (n_kv[k][v] + beta[v]) / (n_kv.index({Slice(k, None)}).sum() + beta.sum());
            }
        }

        weights_ = {varphi, theta};
        return std::make_tuple(z.t().clone(), n_kv, n_mk);
    }

    std::vector<torch::Tensor> get_weights(void) {
    	return weights_;
    }
private:
    int iter_max = 0;
    std::vector<torch::Tensor> weights_;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Set random number generator seed\n";
	std::cout << "// --------------------------------------------------\n";
	std::srand((unsigned) time(NULL));

	GibbsSamplingLDA gibbs_sampling_lda(1000);

    // 输入文本-单词矩阵，共有9个文本，11个单词
	torch::Tensor words = torch::tensor(
					 {{0, 0, 1, 1, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 1, 0, 0, 1},
                      {0, 1, 0, 0, 0, 0, 0, 1, 0},
                      {0, 0, 0, 0, 0, 0, 1, 0, 1},
                      {1, 0, 0, 0, 0, 1, 0, 0, 0},
                      {1, 1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 0, 1, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 1, 0, 1},
                      {0, 0, 0, 0, 0, 2, 0, 0, 1},
                      {1, 0, 1, 0, 0, 0, 0, 1, 0},
                      {0, 0, 0, 1, 1, 0, 0, 0, 0}}, torch::kInt32);
    // 假设话题数量为3
    int K = 3;

    // 设置精度为3
    //np.set_printoptions(precision=3, suppress=True)
    torch::Tensor z, n_kv, n_mk;
    std::tie(z, n_kv, n_mk) = gibbs_sampling_lda.fit(words, K);
	torch::Tensor varphi = gibbs_sampling_lda.get_weights()[0];
	torch::Tensor theta = gibbs_sampling_lda.get_weights()[1];

	std::cout << "文本的话题序列z：\n";
    std::cout << z << '\n';
    std::cout <<  "样本的计数矩阵N_KV：\n";
    std::cout << n_kv << '\n';
    std::cout << "样本的计数矩阵N_MK：\n";
    std::cout << n_mk << '\n';
	std::cout << "模型参数varphi：\n";
	std::cout << varphi << '\n';
	std::cout << "模型参数theta：\n";
	std::cout << theta << '\n';

	std::cout << "Done!\n";
	return 0;
}





