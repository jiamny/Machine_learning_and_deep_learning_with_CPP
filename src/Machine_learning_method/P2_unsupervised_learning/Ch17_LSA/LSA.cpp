/*
 * LSA.cpp
 *
 *  Created on: May 20, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <set>

#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/csvloader.h"

using torch::indexing::Slice;
using torch::indexing::None;


std::string replace_all_char(std::string str, std::string replacement, std::vector<std::string> toBeReplaced) {
	for(auto& toberep : toBeReplaced) {
		//std::replace(str.begin(), str.end(), toberep, replacement);
	    size_t pos;
	    while ((pos = str.find(toberep)) != std::string::npos) {
	        str.replace(pos, 1, replacement);
	    }
	}
	return str;
}

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        if(!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>, std::vector<std::string>> load_data(
		std::string path, std::vector<std::string> string_punctuations, std::vector<std::string> english_stopword) {

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
	return std::make_tuple(topics, text, words);
}

//定义构建单词-文本矩阵的函数，这里矩阵的每一项表示单词在文本中的出现频次，也可以用TF-IDF来表示
torch::Tensor frequency_counter(std::vector<std::vector<std::string>> text, std::vector<std::string> words) {
    /*
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表
    OUTPUT:
    X - (array) 单词-文本矩阵
    */
    torch::Tensor X = torch::zeros({static_cast<long int>(words.size()), static_cast<long int>(text.size())});  //定义m*n的矩阵，其中m为单词列表中的单词个数，n为文本个数
    for(auto& i : range(static_cast<int>(text.size()), 0)) {
    	std::vector<std::string> t = text[i];		//读取文本列表中的第i条文本
        for(auto& w : t ) {
        	auto it = std::find(words.begin(), words.end(), w);
        	if (it != words.end()) {
        		int64_t ind = it - words.begin();  	//取出第i条文本中的第t个单词在单词列表中的索引
        		X[ind][i] += 1;  					//对应位置的单词出现频次加一
        	}
        }
    }
    return X;
}

// 定义潜在语义分析函数
std::vector<std::string> do_lsa(torch::Tensor X, int64_t k, std::vector<std::string> words) {

	torch::Tensor w, v, sort_inds, _;
    std::tie(w, v) = torch::linalg::eig(torch::matmul(X.t(), X));  // 计算Sx的特征值和特征向量，其中Sx=X.T*X，Sx的特征值w即为X的奇异值分解的奇异值，v即为对应的奇异向量
    w = w.to(X.dtype());
    v = v.to(X.dtype());

    torch::Tensor index = w.argsort(-1, true);
	w = torch::index_select(w, 0, index );
	v = torch::index_select(v, 1, index );
	for(int i = 0; i < index.size(0); i++) {
		float  n = torch::norm(v.index({index[i], Slice()})).data().item<float>();
		v.index_put_({index[i], Slice()}, v.index({index[i], Slice()})/n);
	}

	torch::Tensor Sigma = torch::diag(torch::sqrt(w)).to(X.dtype());			// 将特征值数组w转换为对角矩阵，即得到SVD分解中的Sigma
	torch::Tensor U = torch::zeros({static_cast<int64_t>(words.size()), k}, X.dtype()); // 用来保存SVD分解中的矩阵U
    for(int64_t i = 0; i < k; i++ ) {
    	torch::Tensor ui = torch::matmul(X, v.t().index({Slice(), i})) / Sigma[i][i];  // 计算矩阵U的第i个列向量
        //U[:, i] = ui  // 保存到矩阵U中
        U.index_put_({Slice(), i}, ui);
    }

    //topics = []  #用来保存k个话题
    std::vector<std::string> topics;
    for( int64_t i = 0; i < k; i++ ) {
        //inds = np.argsort(U[:, i])[::-1]  // U的每个列向量表示一个话题向量，话题向量的长度为m，其中每个值占向量值之和的比重表示对应单词在当前话题中所占的比重，这里对第i个话题向量的值降序排列后取出对应的索引值
    	torch::Tensor inds = U.index({Slice(), i}).argsort(-1, true);
    	std::string topic = "";  // 用来保存第i个话题
        for(int j = 0; j < 10; j++)
            topic = topic + (" " + words[inds[j].data().item<int>()]);  // 根据索引inds取出当前话题中比重最大的10个单词作为第i个话题
        topics.push_back(topic);  //保存话题i
    }
    return topics;
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

	std::string path = "./data/bbc_text.csv";
	std::vector<std::string> topics, words;
	std::vector<std::vector<std::string>> text;
	std::tie(topics, text, words) = load_data(path, string_punctuations, english_stopword);

	for(auto& t : topics)
		std::cout << t << " ";
	std::cout << '\n';
	std::cout <<"topics: " << topics.size() << '\n';
	std::cout <<"text: "   << text.size() << '\n';
	std::cout <<"words: "  << words.size() << '\n';

	torch::Tensor X = frequency_counter(text, words);		// 构建单词-文本矩阵
	int k = topics.size();  								// 设定话题数为5

	std::vector<std::string> tops = do_lsa(X, k, words);	// 进行潜在语义分析
	std::cout << "Generated Topics:\n";
	for(auto& i : range(k, 0) ) {
		printf("Topic %d: %s\n", i+1, tops[i].c_str());		// 打印分析后得到的每个话题
	}

	std::cout << "Done!\n";
	return 0;
}


