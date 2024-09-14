/*
 * apriori.cpp
 *
 *  Created on: Aug 25, 2024
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
#include <set>
#include <map>

#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class Apriori {
public:
	Apriori() {}

	std::vector<std::vector<std::string>> create_c1( std::vector<std::vector<std::string>> data) {  // 遍历整个数据集生成c1候选集
        std::vector<std::vector<std::string>> c1;
        for(auto& i : data) {
            for(auto& j : i ) {
            	bool hasItem = false;
            	for(auto& c : c1) {
            		auto p = std::find(c.begin(), c.end(), j);
            		if( p != c.end()) {
            			hasItem = true;
            			break;
            		}
            	}

            	if( ! hasItem ) {
            		std::vector<std::string> new_s;
            		new_s.push_back(j);
            		c1.push_back(new_s);
            	}
            }
        }
        return c1;
    }

	// 通过候选项ck生成lk，并将各频繁项的支持度保存到support_data字典中
	std::vector<std::string> generate_lk_by_ck(std::vector<std::vector<std::string>> data,
			std::vector<std::vector<std::string>> ck, int min_support, std::map<std::string, int>& support_data){
        std::map<std::string, int> item_count;  // 用于标记各候选项在数据集出现的次数
        std::vector<std::string> Lk;
        //for(auto& t : data) {
        for(auto& item : ck){
        	for(auto& t : data) {
        		if( isSubset(t, item) ) {
        			std::string itm = join(item, "_");
        			if( item_count.find(itm) == item_count.end() )
        				item_count[itm] = 1;
        			else
        			 	item_count[itm] += 1;
        		}
        	}
        }

        int t_num = data.size();
        for(auto it = item_count.begin(); it != item_count.end(); it++ ) {
        	std::string item = it->first;

        	if( item_count[item] >= min_support) {
        		Lk.push_back(item);
                support_data[item] = item_count[item];
            } else {

            }
        }
        std::cout << "Lk_1: " << Lk.size() << " item_count: " << item_count.size() << '\n';
        return Lk;
    }

	std::vector<std::vector<std::string>> create_ck(std::vector<std::string> Lk, int size) {  // 通过频繁项集Lk-1创建ck候选项集

	    std::vector<std::string> Lk_1 = Lk;
		std::vector<std::vector<std::string>> Ck;

	    int l = Lk_1.size();
	    for(auto& i : range(l, 0)) {
	        for(int j = i + 1; j < l; j++) {  // 两次遍历Lk-1，找出前n-1个元素相同的项
	        	std::vector<std::string> l1 = stringSplit(Lk_1[i], '_');
	        	std::vector<std::string> l2 = stringSplit(Lk_1[j], '_');
	            std::sort(l1.begin(), l1.end());
	            std::sort(l2.begin(), l2.end());
	            std::vector<std::string> s_l1(l1.begin() + 0, l1.begin() + (size - 2));
	            std::vector<std::string> s_l2(l2.begin() + 0, l2.begin() + (size - 2));
	            std::string s1 = "", s2 = "";
	            if( s_l1.size() > 0 ) s1 = join(s_l1, "_");
	            if( s_l2.size() > 0 ) s2 = join(s_l2, "_");

	            if( s1 == s2 ) {  // 只有最后一项不同时，生成下一候选项
	                //Ck_item = lk_list[i] | lk_list[j]
	            	std::vector<std::string> u(l1.size() + l2.size());
	            	std::vector<std::string>::iterator it, st;
	            	std::vector<std::string> m;
	            	// union the vectors using the set_union function
	            	it = std::set_union(l1.begin(), l1.end(), l2.begin(), l2.end(), u.begin());
	            	for(st = u.begin(); st != it; st++)
	            		m.push_back(*st);
	            	std::sort(m.begin(), m.end());

					if( has_infrequent_subset(m, Lk_1) ) {  // 检查该候选项的子集是否都在Lk-1中
	                    Ck.push_back(m);
					}
	            }
	        }
	    }
	    std::cout << "Ck: " << Ck.size() << '\n';
        return Ck;
	}

    bool has_infrequent_subset(std::vector<std::string> merged, std::vector<std::string> Lk_1) {  // 检查候选项Ck_item的子集是否都在Lk-1中
        for(int i = 0; i < merged.size(); i++) {
        	std::vector<std::string> sub_Ck;
        	for(int j = 0; j < merged.size(); j++) {
        		if( j != i )
        			sub_Ck.push_back(merged[j]);
        	}

        	bool hasItem = false;
        	for(auto& t : Lk_1) {
        		std::vector<std::string> tt = stringSplit(t, '_');
        		if( isSubset(tt, sub_Ck) ) {
        			hasItem = true;
        			break;
        		}
        	}

        	if( ! hasItem ) {
        		return false;
        	}
        }
        return true;
    }

    std::tuple<std::vector<std::vector<std::string>>, std::map<std::string, int>> generate_L(
    							std::vector<std::vector<std::string>> data, int min_support) {  // 用于生成所有频繁项集的主函数，k为最大频繁项的大小
    	std::map<std::string, int> support_data;  // 用于保存各频繁项的支持度
    	std::vector<std::vector<std::string>> C1 = create_c1(data);
    	std::vector<std::string> L1 = generate_lk_by_ck(data, C1, min_support, support_data);  // 根据C1生成L1
    	std::vector<std::string> Lksub1 = L1;  // 初始时Lk-1=L1
    	std::vector<std::vector<std::string>> L;
        L.push_back(Lksub1);
        int i = 2;
        while(true) {
        	std::cout << " i ======================== " << i << " support_data: " << support_data.size() << "\n";
		    std::cout << Lksub1[0] << '\n';
		    std::vector<std::vector<std::string>>  Ci = create_ck(Lksub1, i);						// 根据Lk-1生成Ck
		    std::vector<std::string> Li = generate_lk_by_ck(data, Ci, min_support, support_data);	// 根据Ck生成Lk
            if(Li.size() == 0) break;
            Lksub1 = Li;																			// 下次迭代时Lk-1=Lk
            L.push_back(Lksub1);
            i += 1;
        }
        for(auto& i : range(static_cast<int>(L.size()), 0)) {
            printf("frequent item %d：%ld\n", (i + 1), L[i].size());
        }

        return std::make_tuple(L, support_data);
    }

    std::vector<std::pair<std::string, double>> generate_R(std::vector<std::vector<std::string>>  dataset,
    														int min_support, double min_conf) {
    	std::vector<std::vector<std::string>> L;
    	std::map<std::string, int> support_data;
        auto rt = generate_L(dataset, min_support);  // 根据频繁项集和支持度生成关联规则
        L = std::get<0>(rt);
		support_data = std::get<1>(rt);

        std::map<std::string, double> rule_list;  // 保存满足置信度的规则
        std::vector<std::string> sub_set_list;  // 该数组保存检查过的频繁项

        for(auto& i : range(static_cast<int>(L.size()), 0)) {
        	int cnt = 0;
            for(auto& freq_set : L[i] ) {	// 遍历Lk

                for(auto& sub_set : sub_set_list) {	// sub_set_list中保存的是L1到Lk-1
                	std::vector<std::string> tk_freq_set = stringSplit(freq_set, '_');
                	std::vector<std::string> tk_sub_set = stringSplit(sub_set, '_');

                    if( isSubset( tk_freq_set, tk_sub_set )) {	// 检查sub_set是否是freq_set的子集
                        // 检查置信度是否满足要求，是则添加到规则
                    	std::vector<std::string> t = subVector(tk_freq_set, tk_sub_set);
                    	std::sort(t.begin(), t.end());
                    	std::string st = join(t, "_");

                        double conf = support_data[freq_set]*1.0 / support_data[st]; //freq_set - sub_set]
                        std::string big_rule = st + "=" + sub_set;

                        if( conf >= min_conf ) {
                        	if( rule_list.find(big_rule) == rule_list.end() )
                        		rule_list[big_rule] = conf;
                        }

                    }
                }
                cnt++;
                if(cnt % 100 == 0 )
                	std::cout  << "Lk_" << i << " processed " << cnt << "/" << L[i].size() << '\n';
                sub_set_list.push_back(freq_set);
            }
        }

        std::vector<std::pair<std::string, double>> arr(rule_list.begin(), rule_list.end());

        std::sort(arr.begin(), arr.end(),
                      [](const auto &x, const auto &y) { return x.second > y.second; });
        return arr;
    }
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load data\n";
	std::cout << "// --------------------------------------------------\n";

	std::string file_name = "./data/groceries.csv";
	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<std::vector<std::string>> data;

	if( fL.is_open() ) {
		while ( std::getline(fL, line) ) {
			//line = std::regex_replace(line, std::regex("\\\n"), "");
			line = strip(line);
			std::vector<std::string> strs = stringSplit(line, ',');
			std::sort(strs.begin(), strs.end());
			data.push_back(strs);
		}
	}
	fL.close();

	std::cout << "data: " << data.size() << '\n';
    int min_support = 15;	// 最小支持度
    double min_conf = 0.7;	// 最小置信度

    Apriori apriori;
    std::vector<std::pair<std::string, double>> rlt = apriori.generate_R(data, min_support, min_conf);

    std::string filename("./src/Machine_learning_method/P4_others/Apriori/groceries_apriori.txt");
    fstream file_out;

    file_out.open(filename, std::ios_base::out);
    if( ! file_out.is_open() ) {
    	std::cout << "Failed to open " << filename << '\n';
    } else {
    	file_out << "index\tconfidence\trules" << endl;
    	for(int i = 0; i < rlt.size(); i++ ) {
    		std::pair<std::string, double> it = rlt[i];
    		std::vector<std::string> st = stringSplit(it.first, '=');
    		std::string input = st[0];
    	    // Find the first occurrence of the substring
    	    size_t pos = input.find("_");

    	    // Iterate through the string and replace all occurrences
    	    while (pos != string::npos) {
    	        // Replace the substring with the specified string
    	        input.replace(pos, 1, ", ");
    	        // Find the next occurrence of the substring
    	        pos = input.find("_", pos + 1);
    	    }
    		file_out << (i+1) << "\t" << it.second << "\t[" << input << "] => [" << st[1] << "]" << endl;
    	}
    }
    file_out.close();
/*
    std::vector<std::string> a = {"frankfurter", "pip fruit", "onions", "whole milk",
    		"curd", "yogurt", "pastry", "Instant food products", "kitchen utensil"};
    std::vector<std::string> b = {"kitchen utensil"};
    std::vector<std::string> c(a.size() + b.size());
    vector<std::string>::iterator it, st;
    it = std::set_union(a.begin(), a.end(), b.begin(), b.end(), c.begin());
    std::vector<std::string> t;
    for(st = c.begin(); st != it; ++st)
    	t.push_back(*st);
    printVector(t);
*/
	std::cout << "Done!\n";
	return 0;
}



