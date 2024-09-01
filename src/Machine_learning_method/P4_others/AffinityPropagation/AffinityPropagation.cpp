/*
 * viterbi_algorithm.cpp
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

#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

torch::Tensor distance(torch::Tensor point_1, torch::Tensor point_2,
		std::string method = "euclidean", int p=2) {
    if( method == "euclidean" )
        return torch::norm(point_1 - point_2, 2);
    else if(method == "manhattan")
        return torch::sum(torch::abs(point_1 - point_2));
    else if( method == "minkowski" ) {
    	torch::Tensor t = torch::pow(torch::abs(point_1.sub(point_2)), p);
        return torch::pow(torch::sum(t), 1.0/p);
    } else {
        std::cout << "Unknown similarity distance type\n";
        return torch::empty(0);
    }
}

torch::Tensor squareform_dist(torch::Tensor X, std::string method = "euclidean", int p = 2) {
	int size = X.size(0);
	torch::Tensor dist = torch::zeros({size, size}, torch::kFloat32);
	for(int i = 0; i < (size -1); i++) {
		for(int j = 1; j < size; j++) {
			if( i != j ) {
				float d = distance(X[i], X[j], method).data().item<float>();
				dist[i][j] = d;
				dist[j][i] = d;
			}
		}
	}
	return dist;
}

class AffinityPropagation {
public:
	AffinityPropagation(torch::Tensor similariy_matrix, int _max_iteration=200,
						float _alpha=0.5, int _print_every=100) {
        /*
        :param similariy_matrix:
        :param max_iteration:
        :param num_iter:
        :param alpha:
        :param print_every:
        */
        s = similariy_matrix;
        max_iteration = _max_iteration;
        alpha = _alpha;
        print_every = _print_every;
        //N, N = self.s.shape
        r = torch::zeros(s.sizes(), torch::kFloat32);
        a = torch::zeros(s.sizes(), torch::kFloat32);
	}

    std::tuple<torch::Tensor, torch::Tensor> step(void) {
        /*
        :param r is responsiblity matrix, For each data point x_i, how well-suited is x_k as it exempler along with
        other exemplars.
        :param a is availability matrix, For appropriate is x_k as exemplers for x_i, while keeping other data points
        who keeps x_k as exemplar.
        :return:
        */

        int N = s.size(0);
        torch::Tensor old_r = r.clone();
        torch::Tensor old_a = a.clone();
        torch::Tensor a_plus_s = a + s;

        torch::Tensor first_max, first_max_indices;
        std::tie(first_max, first_max_indices) = torch::max(a_plus_s, 1);

        //torch::Tensor first_max_indices = torch::argmax(a_plus_s, dim);
        std::optional<long int> dm = {0};
        first_max = torch::repeat_interleave(first_max, N, dm).reshape({N, N});

        torch::Tensor idx = torch::tensor(range(N, 0)).to(torch::kLong);
        //a_plus_s[range(N), first_max_indices] = float('-inf')
        a_plus_s.index_put_({idx, first_max_indices}, -INFINITY);
        torch::Tensor second_max, _;
        std::tie(second_max, _) = torch::max(a_plus_s, 1);

        // responsibility Update
        r = s - first_max;
        r.index_put_({Slice(0,N), first_max_indices}, s.index({Slice(0,N), first_max_indices}) - second_max);
        r = alpha * old_r + (1 - alpha) * r;

        torch::Tensor rp = torch::maximum(r, torch::zeros(r.sizes()));

        int m = rp.size(0);

        //rp.as_strided([m], [m + 1]).copy_(torch.diag(r))
        rp.index_put_({torch::tensor(range(m, 0)), torch::tensor(range(m, 0))}, torch::diag(r));

        c10::OptionalArrayRef<long int> dd = {0};
        torch::Tensor ss = torch::sum(rp, dd);
        a = torch::repeat_interleave(ss, N, dm).reshape({N, N}).t() - rp;

        torch::Tensor da = torch::diag(a);
        a = torch::minimum(a, torch::zeros(a.sizes()));
        int k = a.size(0);

        //a.as_strided([k], [k+1]).copy_(da)
        a.index_put_({torch::tensor(range(k, 0)), torch::tensor(range(k, 0))}, da);
        // Availibility Update
        a = alpha * old_a + (1 - alpha) * a;

        return std::make_tuple(r, a);
    }

    std::tuple<torch::Tensor, torch::Tensor> solve(void) {
        for(auto& i : range(max_iteration, 0)) {
            std::tie(r, a) = step();

            if( print_every > 0) {
            	if( (i + 1) % print_every == 0 )
            		std::cout << "Iteration: " << (i+1) << '\n';
            }
        }

        torch::Tensor e = r + a;

        //int N = e.size(0);
        torch::Tensor exemplar_indices = torch::where(torch::diag(e) > 0)[0];
        int K = exemplar_indices.size(0);

        torch::Tensor c = s.index({Slice(), exemplar_indices});
        std::optional<long int> dim = {1};
        c = torch::argmax(c, dim);
        c.index_put_({exemplar_indices}, torch::arange(0, K)); //[exemplar_indices] = torch::arange(0, K);

        int N = c.size(0);
        torch::Tensor exemplar_assignment = torch::zeros(N, torch::kInt32);//exemplar_indices[c]
        for(auto& i : range(N, 0)) {
        	int idx = c[i].data().item<int>();
        	exemplar_assignment[i] = exemplar_indices[idx].data().item<int>();
        }
        return std::make_tuple(exemplar_indices, exemplar_assignment);
    }

private:
	torch::Tensor s, r, a;
	int max_iteration = 0, print_every = 0;
	float alpha = 0.;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";

	std::ifstream file;
	std::string path = "./data/breast_cancer.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	//num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> iMap;
    iMap.insert({"malignant", 0});
    iMap.insert({"benign", 1});

    std::cout << "iMap['benign']: " << iMap["benign"] << '\n';

	torch::Tensor data, labels;
	std::tie(data, labels) = process_data2(file, iMap, false, false, false);

	file.close();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	data = data.index_select(0, sidx);
	labels = labels.index_select(0, sidx);
	std::cout << "data: " << data.sizes() << '\n';

	torch::Tensor similarity_matrix = squareform_dist(data, "euclidean");
	std::cout << "dist[):10, 0:10] "
			  << similarity_matrix.index({Slice(0, 10), Slice(0, 10)}) << "\n"
			  << similarity_matrix.sizes() << '\n';

/*
	torch::Tensor a = torch::arange(0, 100).reshape({10, 10});
	torch::Tensor first_max, idx;
	std::tie(first_max, idx)= torch::max(a, 1);
	std::cout << "first_max: " << first_max << '\n' << idx << '\n';
	std::optional<long int> dim = {1};
	torch::Tensor first_max_indices = torch::argmax(a, dim);
	std::cout << "first_max_indices: " << first_max_indices << '\n';
	std::optional<long int> dm = {0};
	first_max = torch::reshape(torch::repeat_interleave(first_max, 10, dm), {10, 10});
	std::cout << "first_max: " << first_max << '\n';

	c10::OptionalArrayRef<long int> dd = {0};
	torch::Tensor ss = torch::sum(a, dd);
	std::cout << "torch::sum(a, dd): " << ss << '\n';
	torch::Tensor b = torch::repeat_interleave(ss, 10, dm).reshape({10, 10}).t();
	std::cout << "b: " << b << '\n';


	std::cout << torch::diag(a).sizes() << '\n' << torch::diag(a) << '\n';
	torch::Tensor rp = a.clone();
	int m = rp.size(0);
	at::IntArrayRef sz1 = {m, m};
	at::IntArrayRef sz2 = {m, m};
	std::cout << "rp:\n" << rp << '\n';
	//std:cout << at::as_strided(rp, sz1, sz2).copy_(torch::diag(a)) << '\n';

	rp.index_put_({torch::tensor(range(m, 0)), torch::tensor(range(m, 0))}, torch::diag(a)*10);
	std::cout << "rp - 2:\n" << rp << '\n';
	std::cout << "torch::arange(0, 2): " << torch::arange(0, 2) << '\n';
*/
	int max_iteration = 5000;
	AffinityPropagation affinity_prop(similarity_matrix, max_iteration, 0.5, 500);
	torch::Tensor indices, assignment;
	std::tie(indices, assignment) = affinity_prop.solve();

	std::map<int, int> iidx;
	for(int i = 0; i < indices.size(0); i++) {
		int idx = indices[i].data().item<int>();
		iidx[idx] = labels[idx].data().item<int>();
		std::cout << "indices[i] = " << idx  << " labels[idx] = " << labels[idx] << '\n';
	}

	int correct = 0;
	for(int i = 0; i < assignment.size(0); i++) {
		int asidx = assignment[i].data().item<int>();
		int label = iidx[asidx];
		int trueLabel = labels[i].data().item<int>();
		if( label == trueLabel)
			correct++;
	}

	std::cout << "Correct assignment = " << correct << " / " << labels.size(0) << '\n';

	std::cout << "Done!\n";
	return 0;
}
