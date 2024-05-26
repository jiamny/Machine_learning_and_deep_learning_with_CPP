/*
 * 9.ExpectationMaximization.cpp
 *
 *  Created on: May 14, 2024
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
#include <cmath>
#include <random>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class GaussianMixture {
public:
	GaussianMixture() {};

	torch::Tensor calcGauss(torch::Tensor dataSetArr, double mu, double sigmod) {
	   /*
	    根据高斯密度函数计算值
	    依据：“9.3.1 高斯混合模型” 式9.25
	    注：在公式中y是一个实数，但是在EM算法中(见算法9.2的E步)，需要对每个j
	    都求一次yjk，在本实例中有1000个可观测数据，因此需要计算1000次。考虑到
	    在E步时进行1000次高斯计算，程序上比较不简洁，因此这里的y是向量，在numpy
	    的exp中如果exp内部值为向量，则对向量中每个值进行exp，输出仍是向量的形式。
	    所以使用向量的形式1次计算即可将所有计算结果得出，程序上较为简洁
	    :param dataSetArr: 可观测数据集
	    :param mu: 均值
	    :param sigmod: 方差
	    :return: 整个可观测数据集的高斯分布密度（向量形式）
	   */

	    //计算过程就是依据式9.25写的，没有别的花样
		torch::Tensor result = (1. / (std::sqrt(2. * M_PI) * sigmod)) *
	             torch::exp(-1. * (dataSetArr - mu) * (dataSetArr - mu) / (2. * sigmod*sigmod));
	    //print(1/(math.sqrt(2 * math.pi) * sigmod))
	    // 返回结果
	    return result;
	}

	std::pair<torch::Tensor, torch::Tensor> E_step(torch::Tensor dataSetArr, double alpha0, double mu0, double sigmod0,
			double alpha1, double mu1, double sigmod1) {
	    /*
	    EM算法中的E步
	     依据当前模型参数，计算分模型k对观数据y的响应度
	    :param dataSetArr: 可观测数据y
	    :param alpha0: 高斯模型0的系数
	    :param mu0: 高斯模型0的均值
	    :param sigmod0: 高斯模型0的方差
	    :param alpha1: 高斯模型1的系数
	    :param mu1: 高斯模型1的均值
	    :param sigmod1: 高斯模型1的方差
	    :return: 两个模型各自的响应度
	    */

	    //计算y0的响应度
	    //先计算模型0的响应度的分子
		torch::Tensor gamma0 = alpha0 * calcGauss(dataSetArr, mu0, sigmod0);

	    //模型1响应度的分子
		torch::Tensor gamma1 = alpha1 * calcGauss(dataSetArr, mu1, sigmod1);

	    //两者相加为E步中的分布
		torch::Tensor sum = gamma0 + gamma1;
	    // 各自相除，得到两个模型的响应度
	    gamma0 = gamma0 / sum;
	    gamma1 = gamma1 / sum;

	    //返回两个模型响应度
	    return std::make_pair(gamma0, gamma1);
	}

	std::tuple<double, double, double, double, double, double> M_step(
			double muo, double mu1, torch::Tensor gamma0, torch::Tensor gamma1, torch::Tensor dataSetArr) {
	    //依据算法9.2计算各个值
	    //这里没什么花样，对照书本公式看看这里就好了
		double mu0_new = (torch::dot(gamma0, dataSetArr) / torch::sum(gamma0)).data().item<double>();
		double mu1_new = (torch::dot(gamma1, dataSetArr) / torch::sum(gamma1)).data().item<double>();

		double sigmod0_new = (torch::sqrt(torch::dot(gamma0,
				torch::pow((dataSetArr - muo), 2)) / torch::sum(gamma0))).data().item<double>();

		double sigmod1_new = (torch::sqrt(torch::dot(gamma1,
				torch::pow((dataSetArr - mu1),2)) / torch::sum(gamma1))).data().item<double>();

		double alpha0_new = (torch::sum(gamma0) / gamma0.size(0)).data().item<double>();
		double alpha1_new = (torch::sum(gamma1) / gamma1.size(0)).data().item<double>();

	    //将更新的值返回
	    return std::make_tuple(mu0_new, mu1_new, sigmod0_new, sigmod1_new, alpha0_new, alpha1_new);
	}

	std::tuple<double, double, double, double, double, double> fit(torch::Tensor dataSetArr, int iter = 500) {
	   /*
	    根据EM算法进行参数估计
	    算法依据“9.3.2 高斯混合模型参数估计的EM算法” 算法9.2
	    :param dataSetList:数据集（可观测数据）
	    :param iter: 迭代次数
	    :return: 估计的参数
	   */
	    //将可观测数据y转换为数组形式，主要是为了方便后续运算

	    //步骤1：对参数取初值，开始迭代
	    double alpha0 = 0.5, mu0 = 0, sigmod0 = 1,
	    alpha1 = 0.5, mu1 = 1, sigmod1 = 1;

	    //开始迭代
	    int step = 0;
	    torch::Tensor  gamma0, gamma1;
	    while (step < iter) {
	        //每次进入一次迭代后迭代次数加1
	        step += 1;
	        //步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度
	        std::tie(gamma0, gamma1) = E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);
	        //步骤3：M步
	        std::tie(mu0, mu1, sigmod0, sigmod1, alpha0, alpha1) =
	            M_step(mu0, mu1, gamma0, gamma1, dataSetArr);

	        if( step % 10 == 0) {
	        	printf("Iter step %3d => alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f\n",
	        			step, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);
	        }
	    }

	    //迭代结束后将更新后的各参数返回
	    return std::make_tuple(alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);
	}

};


torch::Tensor loadData(double mu0, double sigma0, double mu1, double sigma1, double alpha0, double alpha1) {
   /*
    初始化数据集
    这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
   */

    //定义数据集长度为1000
    int length = 1000;

    //初始化总数据集
    //两个高斯分布的数据混合后会放在该数据集中返回
    std::vector<torch::Tensor> dt;
    torch::Tensor data0 = torch::normal(mu0, sigma0, {int(length * alpha0)});
    dt.push_back(data0);
    //第二个高斯分布的数据
    torch::Tensor data1 = torch::normal(mu1, sigma1, {int(length * alpha1)});
    dt.push_back(data1);
    torch::Tensor dataSet = torch::cat(dt, -1);

    std::cout << dataSet.sizes() << '\n';

    //返回伪造好的数据集
    return dataSet;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

    double alpha0 = 0.3, mu0 = -2, sigmod0 = 0.5,
    alpha1 = 0.7, mu1 = 0.5, sigmod1 = 1;

    //初始化数据集
    torch::Tensor dataset = loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1);
    GaussianMixture gau = GaussianMixture();

    // 打印设置的参数
    printf("---------------------------\n");
    printf("the Parameters set is:\n");
    printf("alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f\n",
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);

    //开始EM算法，进行参数估计
    std::tie(alpha0, mu0, sigmod0, alpha1, mu1, sigmod1) = gau.fit(dataset);

	//打印参数预测结果
	printf("---------------------------\n");
    printf("the Parameters predict is:\n");
	printf("alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f\n",
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);

	std::cout << "Done!\n";
	return 0;
}


