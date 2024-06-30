/*
 * LDA.cpp
 *
 *  Created on: Jun 11, 2024
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

// 定义构建单词-文本矩阵的函数，这里矩阵的每一项表示单词在文本中的出现频次，也可以用TF-IDF来表示
std::tuple<std::vector<std::vector<std::string>>, std::vector<std::string>, torch::Tensor>
			frequency_counter(std::vector<std::vector<std::string>> text, std::vector<std::string> words, int M) {
    /*
    text - (list) 文本列表
    words - (list) 单词列表

    OUTPUT:
    words - (list) 出现频次为前1000的单词列表
    X - (array) 单词-文本矩阵
    */
    torch::Tensor words_cnt = torch::zeros({static_cast<long int>(words.size())}, torch::kInt32);   // 用来保存单词的出现频次
    // beta是LDA模型的另一个超参数，是词汇表中单词的概率分布，这里取各单词在所有文本中的比例作为beta值，实际也可以通过模型训练得到
    torch::Tensor beta = torch::zeros({M}, torch::kDouble);

    // 循环计算words列表中各单词出现的词频
    for(auto& i : range(static_cast<int>(text.size()), 0)) {
    	std::vector<std::string> t = text[i];  // 取出第i条文本
        for(auto& w : t) {
            //ind = words.index(w)  // 取出第i条文本中的第t个单词在单词列表中的索引
            //words_cnt[ind] += 1   // 对应位置的单词出现频次加一
        	auto it = std::find(words.begin(), words.end(), w);
        	if (it != words.end()) {
        		int64_t ind = it - words.begin();  	// 取出第i条文本中的第t个单词在单词列表中的索引
        		words_cnt[ind] += 1;  				// 对应位置的单词出现频次加一
        	}
        }
    }

    // 对单词出现频次降序排列后取出其索引值
    torch::Tensor sort_inds = torch::argsort(words_cnt,  -1, true);
    //sort_inds = torch::flip(sort_inds, {-1});

    // 将出现频次前1000的单词保存到words列表
    std::vector<std::string> wds;
	for(auto& ind : range(M, 0))	 {
		int idx = sort_inds[ind].data().item<int>();
		wds.push_back(words[idx]);
	}

	// 去除文本text中不在词汇表words中的单词
    for(auto& i : range(static_cast<int>(text.size()), 0)) {
        std::vector<std::string> t;
        for(auto& w : text[i] ) {
        	auto it = std::find(wds.begin(), wds.end(), w);
        	if (it != wds.end()) {
        		int64_t ind = it - wds.begin();
                t.push_back(w);
                beta[ind] += 1;  //统计各单词在文本中的出现频次
            }
        }
        text[i] = t;
    }
    beta /= torch::sum(beta);  // 除以文本的总单词数得到各单词所占比例，作为beta值
    return std::make_tuple(text, wds, beta);
}


//定义潜在狄利克雷分配函数，采用收缩的吉布斯抽样算法估计模型的参数theta和phi
std::pair<torch::Tensor, torch::Tensor> do_lda(std::vector<std::vector<std::string>> text,
		std::map<std::string, int> m_words, torch::Tensor alpha, torch::Tensor beta, int K, int iters) {
    /*
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表
    alpha - (list) 话题概率分布，模型超参数
    beta - (list) 单词概率分布，模型超参数
    K - (int) 设定的话题数
    iters - (int) 设定的迭代次数

    OUTPUT:
    theta - (array) 话题的条件概率分布p(zk|dj)，这里写成p(zk|dj)是为了和PLSA模型那一章的符号统一一下，方便对照着看
    phi - (array) 单词的条件概率分布p(wi|zk)
    */

    int M = text.size();  // 文本数
    int V = m_words.size();  // 单词数
    torch::Tensor N_MK = torch::zeros({M, K}, torch::kInt32);   // 文本-话题计数矩阵
    torch::Tensor N_KV = torch::zeros({K, V}, torch::kInt32);   // 话题-单词计数矩阵
    torch::Tensor N_M = torch::zeros({M}, torch::kInt32);  	 // 文本计数向量
    torch::Tensor N_K = torch::zeros({K}, torch::kInt32);		 // 话题计数向量
    std::vector<std::vector<int>> Z_MN;  // 用来保存每条文本的每个单词所在位置处抽样得到的话题

    // 算法20.2的步骤(2)，对每个文本的所有单词抽样产生话题，并进行计数
    for(auto& m : range(M, 0)) {
    	std::vector<int> zm;
        std::vector<std::string> t = text[m];
        for(int n = 0; n < t.size(); n++) {
        	int v = m_words[t[n]];
        	int z = torch::randint(0, K, {1}).data().item<int>();
            zm.push_back(z);
			N_MK[m][z] += 1;
			N_M[m] += 1;
			N_KV[z][v] += 1;
			N_K[z] += 1;
        }
        Z_MN.push_back(zm);
    }

	// 算法20.2的步骤(3)，多次迭代进行吉布斯抽样
    for(auto& i : range(iters, 0)) {
    	int cnt = 0;
        for(auto& m : range(M, 0)) {
            std::vector<std::string> t = text[m];
            for(int n = 0; n < t.size(); n++) {
            	int v = m_words[t[n]];
            	int z = (Z_MN[m])[n];
            	N_MK[m][z] -= 1;
            	N_M[m] -= 1;
            	N_KV[z][v] -= 1;
            	N_K[z] -= 1;
            	// 用来保存对K个话题的条件分布p(zi|z_i,w,alpha,beta)的计算结果
            	torch::Tensor p = torch::zeros({K}, torch::kDouble);
            	double sums_k = 0;
                for(auto&  k : range(K, 0)) {
                    // 话题zi=k的条件分布p(zi|z_i,w,alpha,beta)的分子部分
                    double p_zk = ((N_KV[k][v] + beta[v]) * (N_MK[m][k] + alpha[k])).data().item<double>();
                    double sums_v = 0;
                    sums_k += (N_MK[m][k] + alpha[k]).data().item<double>() ; // 累计(nmk + alpha_k)在K个话题上的和
                    for(auto& t : range(V, 0)) {
                        sums_v += (N_KV[k][t] + beta[t]).data().item<double>();  // 累计(nkv + beta_v)在V个单词上的和
                    }
                    p_zk /= sums_v;
                    p[k] = p_zk;
                }
                p.div_(sums_k);
                p.div_(torch::sum(p));  //对条件分布p(zi|z_i,w,alpha,beta)进行归一化，保证概率的总和为1
                int new_z = static_cast<int>(random_choice(1, tensorTovector(p))[0]);  //根据以上计算得到的概率进行抽样，得到新的话题
                (Z_MN[m])[n] = new_z;  // 更新当前位置处的话题为上面抽样得到的新话题
                // 更新计数
                N_MK[m][new_z] += 1;
                N_M[m] += 1;
                N_KV[new_z][v] += 1;
                N_K[new_z] += 1;
            }
            cnt++;
            if( cnt % 200 == 0)
            	std::cout << "texts: " << cnt << "/" << M << '\n';
        }
        std::cout << "-------------- 迭代次数: " << (i+1) << " / " << iters << '\n';
    }
    // 算法20.2的步骤(4)，利用得到的样本计数，估计模型的参数theta和phi
	torch::Tensor theta = torch::zeros({M, K}, torch::kDouble);
    torch::Tensor phi = torch::zeros({K, V}, torch::kDouble);
    for(auto& m : range(M,0)) {
        double sums_k = 0;
        for(auto& k : range(K,0)) {
            theta[m][k] = (N_MK[m][k] + alpha[k]).data().item<double>();  // 参数theta的分子部分
            sums_k += (theta[m][k]).data().item<double>();  //累计(nmk + alpha_k)在K个话题上的和，参数theta的分母部分
        }
        theta[m] /= sums_k;  // 计算参数theta
    }

    for(auto& k : range(K,0)) {
        double sums_v = 0;
        for(auto& v : range(V,0)) {
            phi[k][v] = (N_KV[k][v] + beta[v]).data().item<double>();  // 参数phi的分子部分
            sums_v += phi[k][v].data().item<double>();  // 累计(nkv + beta_v)在V个单词上的和，参数phi的分母部分
        }
        phi[k] /= sums_v;  // 计算参数phi
    }

    return std::make_pair(theta, phi);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Set random number generator seed for random_choice()\n";
	std::cout << "// --------------------------------------------------\n";
	std::srand((unsigned) time(NULL));

	std::vector<std::string> string_punctuations = {"!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+",
													",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@",
													"[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"};

	std::vector<std::string> english_stopword = {"i", "me", "my", "myself", "we", "our", "ours",
			"ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
			"yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself",
			"it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
			"which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
			"was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
			"doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
			"of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
			"before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
			"over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
			"how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
			"nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
			"just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y",
			"ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
			"hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't",
			"mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn",
			"wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"};

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Load data\n";
	std::cout << "// --------------------------------------------------\n";
	std::string path = "./data/bbc_text.csv";

	std::ifstream file;
	file.open(path, std::ios_base::in);

	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
	}

	std::vector<std::string> topics, texts, _topics;

	std::tie(topics, texts) = process_bbc_news_data(file, true);

	std::copy(topics.begin(), topics.end(), back_inserter(_topics));

	std::sort(topics.begin(), topics.end());
	auto it = std::unique(topics.begin(), topics.end());
	topics.erase(it, topics.end());
	printVector(topics);

	// 计算各话题的比例作为alpha值
	torch::Tensor alpha = torch::zeros({static_cast<int>(topics.size())}, torch::kDouble);
	for(int i = 0; i < topics.size(); i++) {
		int cnt = std::count(_topics.begin(), topics.end(), topics[i]);
    	alpha[i] - cnt*1.0 / topics.size();
	}

	std::vector<std::vector<std::string>> text;
	std::vector<std::string> words;

	for(auto& str: texts) {
		str = replace_all_char(str, "", string_punctuations);
		std::vector<std::string> tokens = stringSplit(str, ' ');
		std::set<std::string> wds;
		std::vector<std::string> tks;
		for(auto& ser : tokens) {
			std::vector<std::string>::iterator it;
			it = std::find(english_stopword.begin(), english_stopword.end(), ser);
			if (it == english_stopword.end()) {
				if(ser.length() > 3) {
					wds.insert(ser);
					tks.push_back(ser);
				}
			}
		}
		text.push_back(tks);

		std::vector<std::string> vec;
		vec.assign(wds.begin(), wds.end());

		for(auto& w : vec)
			words.push_back(w);
	}

	std::set<std::string> vwds(words.begin(), words.end());
	words.assign(vwds.begin(), vwds.end());

	file.close();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// frequency count \n";
	std::cout << "// --------------------------------------------------\n";
	std::vector<std::string> Words;
	std::vector<std::vector<std::string>> Text;
	torch::Tensor beta;
	std::tie(Text,  Words, beta ) = frequency_counter(text, words, 1000);  // 取频次前1000的单词重新构建单词列表，并构建单词-文本矩阵
	printVector(tensorTovector(beta));
	std::cout << Words.size() << '\n';

	std::map<std::string, int> m_words;
	for(int i = 0; i < Words.size(); i++)
		m_words[Words[i]] = i;

	m_words.size();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Do latent Dirichlet allocation \n";
	std::cout << "// --------------------------------------------------\n";
	int K = topics.size(); // 设定话题数为5
	int iters = 1;

	torch::Tensor theta, phi;
	std::tie(theta, phi) = do_lda(Text, m_words, alpha, beta, K, iters); // LDA的吉布斯抽样
	// 打印出每个话题zk条件下出现概率最大的前10个单词，即P(wi|zk)在话题zk中最大的10个值对应的单词，作为对话题zk的文本描述
    for(auto& k : range(K, 0)) {
    	// 对话题zk条件下的P(wi|zk)的值进行降序排列后取出对应的索引值
    	torch::Tensor sort_inds = torch::argsort(phi[k], -1, true);

        std::vector<std::string> topic;  //定义一个空列表用于保存话题zk概率最大的前10个单词
        // 定义一个空列表用于保存话题zk概率最大的前10个单词
        for(auto& i : range(10, 0)) {
        	topic.push_back(Words[sort_inds[i].data().item<int>()]);
        }
        printf("Topic %2d: %s\n", (k+1), join(topic, " ").c_str());  //打印话题
    }

	std::cout << "Done!\n";
	return 0;
}


