/*
 * PLSA.cpp
 *
 *  Created on: Jun 2, 2024
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

#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

// 定义构建单词-文本矩阵的函数，这里矩阵的每一项表示单词在文本中的出现频次，也可以用TF-IDF来表示
std::pair<std::vector<std::string>, torch::Tensor> frequency_counter(std::vector<std::vector<std::string>> text,
															std::vector<std::string> words) {
    /*
    text - (list) 文本列表
    words - (list) 单词列表

    OUTPUT:
    words - (list) 出现频次为前1000的单词列表
    X - (array) 单词-文本矩阵
    */
    torch::Tensor words_cnt = torch::zeros({static_cast<long int>(words.size())}, torch::kInt32);   // 用来保存单词的出现频次
    // 定义m*n的矩阵，其中m为单词列表中的单词个数，为避免运行时间过长，这里只取了出现频次为前1000的单词，因此m为1000，n为文本个数
    torch::Tensor X = torch::zeros({1000, static_cast<long int>(text.size())});  //
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
	for(auto& ind : range(1000, 0))	 {
		int idx = sort_inds[ind].data().item<int>();
		wds.push_back(words[idx]);
	}

    // 构建单词-文本矩阵
    for(auto& i : range(static_cast<int>(text.size()), 0)) {
    	std::vector<std::string> t = text[i];		// 读取文本列表中的第i条文本
        for(auto& w : t ) {
        	auto it = std::find(wds.begin(), wds.end(), w);
        	if (it != wds.end()) {
        		int64_t ind = it - wds.begin();  	// 取出第i条文本中的第t个单词在单词列表中的索引
        		X[ind][i] += 1;  					// 对应位置的单词出现频次加一
        	}
        }
    }
    return std::make_pair(wds, X);
}

// 定义概率潜在语义分析函数，采用EM算法进行PLSA模型的参数估计
std::pair<torch::Tensor, torch::Tensor> do_plsa(torch::Tensor X, int K, std::vector<std::string> words, int iters = 10) {
    /*
    INPUT:
    X - (array) 单词-文本矩阵
    K - (int) 设定的话题数
    words - (list) 出现频次为前1000的单词列表
    iters - (int) 设定的迭代次数

    OUTPUT:
    P_wi_zk - (array) 话题zk条件下产生单词wi的概率数组
    P_zk_dj - (array) 文本dj条件下属于话题zk的概率数组
    */

    int M = X.size(0), N = X.size(1); // M为单词数，N为文本数
    // P_wi_zk表示P(wi|zk)，是一个K*M的数组，其中每个值表示第k个话题zk条件下产生第i个单词wi的概率，这里将每个值随机初始化为0-1之间的浮点数
    torch::Tensor P_wi_zk = torch::rand({K, M});

    // 对于每个话题zk，保证产生单词wi的概率的总和为1
    for(auto& k : range(K, 0)) {
        P_wi_zk[k] /= torch::sum(P_wi_zk[k]);
    }
    // P_zk_dj表示P(zk|dj)，是一个N*K的数组，其中每个值表示第j个文本dj条件下产生第k个话题zk的概率，这里将每个值随机初始化为0-1之间的浮点数
	torch::Tensor P_zk_dj = torch::rand({N, K});

	// 对于每个文本dj，属于话题zk的概率的总和为1
    for(auto& n : range(N, 0)) {
        P_zk_dj[n] /= torch::sum(P_zk_dj[n]);
    }
    // P_zk_wi_dj表示P(zk|wi,dj)，是一个M*N*K的数组，其中每个值表示在单词-文本对(wi,dj)的条件下属于第k个话题zk的概率，这里设置初始值为0
	torch::Tensor P_zk_wi_dj = torch::zeros({M, N, K});

    // 迭代执行E步和M步
    for(auto& i : range(iters, 0)) {

        // 执行E步
        for(auto& m : range(M, 0)) {
            for(auto& n : range(N, 0)) {
                double sums = 0.;
                for(auto& k : range(K, 0)) {
                	// 计算P(zk|wi,dj)的分子部分，即P(wi|zk)*P(zk|dj)
                    P_zk_wi_dj.index_put_({m, n,k},P_wi_zk[k][m] * P_zk_dj[n][k]);
                    // 计算P(zk|wi,dj)的分母部分，即P(wi|zk)*P(zk|dj)在K个话题上的总和
                    sums += P_zk_wi_dj.index({m, n,k}).data().item<double>();
                }
                // 得到单词-文本对(wi,dj)条件下的P(zk|wi,dj)
                auto d = P_zk_wi_dj.index({m, n, Slice()}) / sums;
                P_zk_wi_dj.index_put_({m, n, Slice()}, d);
            }
        }

        printf("Iter: %2d/%2d\t%s\n", (i+1), iters, "E step done.");

        // 执行M步，计算P(wi|zk)
        for(auto& k : range(K, 0)) {
            double s1 = 0.;
            for(auto& m : range(M, 0)) {
                P_wi_zk[k][m] = 0;
                for(auto& n : range(N, 0)) {
                	// 计算P(wi|zk)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和，其中n(wi,dj)为单词-文本矩阵X在文本对(wi,dj)处的频次
                    P_wi_zk[k][m] += (X[m][n] * P_zk_wi_dj.index({m, n, k})).data().item<double>();
                }
                // 计算P(wi|zk)的分母部分，即n(wi,dj)*P(zk|wi,dj)在N个文本和M个单词上的总和
                s1 += (P_wi_zk[k][m]).data().item<double>();
            }
            // 得到话题zk条件下的P(wi|zk)
            auto d = P_wi_zk.index({k, Slice()}) / s1;
            P_wi_zk.index_put_({k, Slice()}, d);
        }

		// 执行M步，计算P(zk|dj)
        for(auto& n : range(N, 0)) {
            for(auto&  k : range(K, 0)) {
                P_zk_dj[n][k] = 0;
                for(auto& m : range(M, 0)) {
					// 同理计算P(zk|dj)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和
                    P_zk_dj[n][k] += (X[m][n] * P_zk_wi_dj.index({m, n, k})).data().item<double>();
                }
                // 得到文本dj条件下的P(zk|dj)，其中n(dj)为文本dj中的单词个数，
                // 由于我们只取了出现频次前1000的单词，所以这里n(dj)计算的是文本dj中在单词列表中的单词数
                P_zk_dj[n][k] = (P_zk_dj[n][k] / torch::sum(X.index({Slice(), n}))).data().item<double>();
            }
        }
        printf("Iter: %2d/%2d\t%s\n", (i+1), iters, "M step done.");
    }
	return std::make_pair(P_wi_zk, P_zk_dj);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

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

	std::vector<std::string> topics, texts;

	std::tie(topics, texts) = process_bbc_news_data(file, true);

	std::sort(topics.begin(), topics.end());
	auto it = std::unique(topics.begin(), topics.end());
	topics.erase(it, topics.end());

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

/*
	torch::Tensor words_cnt = torch::tensor({5, 2, 6, 7, 10});
	torch::Tensor sort_inds = torch::argsort(words_cnt);
	torch::Tensor sort_inds_r = torch::argsort(words_cnt, -1, true);
	printVector(tensorTovector(sort_inds.to(torch::kDouble)));
	sort_inds = torch::flip(sort_inds, {-1});
	printVector(tensorTovector(sort_inds.to(torch::kDouble)));
	printVector(tensorTovector(sort_inds_r.to(torch::kDouble)));
*/
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// frequency count \n";
	std::cout << "// --------------------------------------------------\n";
	std::vector<std::string> Words;
	torch::Tensor X;
	std::tie( Words, X ) = frequency_counter(text, words);  // 取频次前1000的单词重新构建单词列表，并构建单词-文本矩阵
	std::cout << X << '\n';
	std::cout << Words.size() << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Do plsa \n";
	std::cout << "// --------------------------------------------------\n";
	int K = 5; // 设定话题数为5
	torch::Tensor P_wi_zk, P_zk_dj;
	std::tie(P_wi_zk, P_zk_dj) = do_plsa(X, K, words, 10); // 采用EM算法对PLSA模型进行参数估计
    for(auto& k : range(K, 0)) {
    	// 对话题zk条件下的P(wi|zk)的值进行降序排列后取出对应的索引值
    	torch::Tensor sort_inds = torch::argsort(P_wi_zk[k], -1, true);

        std::string topic = "";  //定义一个空列表用于保存话题zk概率最大的前10个单词
        for(auto& i : range(10, 0)) {
        	if( topic.length() > 0)
        		topic += (" " + Words[sort_inds[i].data().item<int>()]);
        	else
        		topic = Words[sort_inds[i].data().item<int>()];
        }
        printf("Topic %2d: %s\n", (k+1), topic.c_str());  //打印话题zk
    }

	std::cout << "Done!\n";
	return 0;
}





