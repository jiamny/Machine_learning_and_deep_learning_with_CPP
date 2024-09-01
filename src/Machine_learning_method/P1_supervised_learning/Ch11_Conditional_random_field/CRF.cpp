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
#include <set>
#include <float.h>
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

bool cmp(std::pair<torch::Tensor, double> a, std::pair<torch::Tensor, double> b) {
	return a.second > b.second;
}

// --------------------------------------------------------------
// 使用条件随机场矩阵形式，计算所有路径状态序列的概率及概率最大的状态序列
// --------------------------------------------------------------
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

// ------------------------------------------------------------------------------------------
// CRF层其实主要包含两部分，一个是计算序列的极大对数似然，也就是深度学习前向传播需要计算的loss，
// 二个是通过学习到的参数通过viterbi算法解码得到最优的标注序列。本文展示前一个模块的前向传播计算。
// ------------------------------------------------------------------------------------------
class CRF : public torch::nn::Module {
    /*
    CRF类
     给定 '标签序列'和'发射矩阵分数' 计算对数似然（也就是损失函数）
     同时有decode函数，通过维特比算法，根据发射矩阵分数计算最优的标签序列（暂不展示）

     超参数：
        num_tags: 标签的个数（一般为自行设计的BIO标签的个数）
     学习参数：
        transitions (torch.nn.Parameter): 转移矩阵， shape=(num_tags, num_tags) 方阵
    */
public:
    CRF(int _num_tags) {
        num_tags = _num_tags;

        // 学习参数
        transitions = torch::empty({num_tags, num_tags});  // (num_tags, num_tags)
        // 重新随机初始化参数矩阵，初始化为-0.1 ~ 0.1的均匀分布
        torch::nn::init::uniform_(transitions, -0.1, 0.1);
        reductions.insert("none");
        reductions.insert("sum");
        reductions.insert("mean");
        reductions.insert("token_mean");
    }

    // 计算给定发射分数张量和真实标签序列，来计算序列的条件对数似然 (conditional log likelihood)
    torch::Tensor forward(torch::Tensor emissions, torch::Tensor tags,
				torch::Tensor mask = torch::empty(0), std::string reduction = "mean") {
        /*
         参数：
            （1）emissions: 发射分数，一般是LSTM/BERT的输出表示,
                           shape = (batch_size, seq_len, num_tags)
            （2）tags:  真实标签序列，每一条BIO类似标注，
                        shape = (batch_size, seq_len)
            （3）mask:  mask矩阵，排除padding的影响，
                        shape = (batch_size, seq_len)
            （4）reduction: 计算最后输出的策略，可选（none|sum|mean|token_mean）。
                       （none什么不做，sum在batch维度求和，mean在batch维度平均，token_mean在token维度平均）
         返回：
            对数似然分数(log likelihood)，如果reduction='none', 则shape=(batch_size,) ，否则返回一个常数
       	*/

        if( reductions.find(reduction) == reductions.end()) {
            // 判断是否非法
            std::cout << "invalid reduction: " << reduction << '\n';
            exit(1);
        }

        if( mask.numel() == 0 ){
            mask = torch::ones_like(tags, torch::kBool).to(tags.device());  //(batch_size, seq_len) 全为1
        }

        if( mask.dtype() != torch::kBool)
            mask = mask.to(torch::kBool);

        // 计算分子
        torch::Tensor numerator = _compute_score(emissions, tags, mask.clone());	// shape=(batch_size,)
        // 计算分母
        torch::Tensor denominator = _compute_normalizer(emissions, mask); 			// shape=(batch_size,)

        torch::Tensor llh = numerator - denominator; // 极大对数似然分数，如果nn反向传播需要加个负号 ，shape=(batch_size)
        if(reduction == "none")
            return llh;
        if(reduction == "sum")
            return llh.sum();	// 常数张量
        if(reduction == "mean")
            return llh.mean();	// 常数张量
        // 下面解释一下：mask.float().sum()代表这一batch里一共有多少个有效字，然后把这些序列的所有字累加。
        // （主要还是去除padding的影响）
        return llh.sum()*1.0 / mask.to(torch::kFloat32).sum();  // 常数张量
    }

    torch::Tensor _compute_score(torch::Tensor emissions, torch::Tensor tags, torch::Tensor mask) {
        /*
        emissions: (batch_size, seq_len, num_tags)
        tags:      (batch_size, seq_len)
        mask:      (batch_size, seq_len)
         返回对数似然的分子： (batch_size)
        */
        int batch_size = tags.size(0), seq_length = tags.size(1);
        mask = mask.to(torch::kFloat32);

        // 序列第一个位置的发射分数，记为score
        torch::Tensor score = emissions.index({torch::arange(batch_size), 0, tags.index({Slice(), 0})});  // (batch_size)

        for(auto& i : range(seq_length - 1, 1)) {  // 对接下来序列的每一个字进行遍历，相加计算总分数
            // 开始累加转移分数，当且仅当mask=1（排除padding）执行
            score += transitions.index({tags.index({Slice(), i-1}), tags.index({Slice(), i})}) * mask.index({Slice(), i}); // （batch_size, )

            // 开始累加发射分数，当且仅当mask=1（排除padding）执行
			score += emissions.index({torch::arange(batch_size), i, tags.index({Slice(), i})}) * mask.index({Slice(), i});
        }

        return score;
    }

    torch::Tensor _compute_normalizer(torch::Tensor emissions, torch::Tensor mask) {
        /*
        emissions: (batch_size, seq_len, num_tags)
        mask: (batch_size, seq_len)
          返回对数似然的分母（归一化因子）：(batch_size,)
        */
        int seq_length = emissions.size(1);

        // 开始发射矩阵的分数，注意到因为要求归一化分母，所以需要知道每个tag的似然分数 (根据公式(5))
        // 即，shape = (batch_size, num_tags)
        torch::Tensor score = emissions.index({Slice(), 0});  // (batch_size, num_tags)

        for(auto& i : range(seq_length - 1, 1)) {
            // 广播score, 方便算每条样本的所有tag之间的转移
        	torch::Tensor broadcast_score = score.unsqueeze(2);   // (batch_size, num_tags, 1)

            // 广播emissions，方便算每条样本的所有tag之间的转移
        	torch::Tensor broadcast_emissions = emissions.index({Slice(), i}).unsqueeze(1);  // (batch_size, 1, num_tags)

            // 对于每个序列，计算仅到此时刻t的归一化因子
        	torch::Tensor next_score = broadcast_score + transitions + broadcast_emissions;  // (bs, num_tags, num_tags)

            // 先指数求和再计算对数，即 logsumexp
        	c10::ArrayRef<long int> dim = {1};
			next_score = torch::logsumexp(next_score, dim);  // (bs, num_tags)

            // 去除padding的影响
			score = torch::where(mask.index({Slice(), i}).unsqueeze(1), next_score, score);
        }
        c10::ArrayRef<long int> dim = {1};
        return torch::logsumexp(score, dim);   // (batch_size,)
    }

private:
    int num_tags = 0;
    torch::Tensor transitions;
    std::set<std::string> reductions;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Tag path\n";
	std::cout << "// --------------------------------------------------\n";
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

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Conditional log likelihood\n";
	std::cout << "// --------------------------------------------------\n";
	auto cf = CRF(5);
	torch::Tensor hid_out = torch::randn({8, 100, 5});
	torch::Tensor y_tag = torch::randint(5, {8, 100});

	torch::Tensor loss = -1.0*cf.forward(hid_out, y_tag);
	std::cout << "loss: " << loss << '\n';

	std::cout << "Done!\n";
	return 0;
}





