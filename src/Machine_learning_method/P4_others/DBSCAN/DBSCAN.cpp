/*
 * DBSCAN.cpp
 *
 *  Created on: Jun 21, 2024
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
#include <vector>
#include <random>
#include <algorithm>
#include <float.h>

#include "../../../Utils/csvloader.h"
#include "../../../Utils/helpfunction.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

/*
 * - Compared to centroid-based clustering like k-means, density-based clustering works by
 * identifying “dense” clusters of points, allowing it to learn clusters of arbitrary shape
 * and identify outliers in the data.
 */

class DBScan {
public:
	DBScan(float _eps = 2.5, int _min_points=30) {
        /*
        eps - radius distance around which a cluster is considered.
        min_points -  Number of points to be present inside the radius
        (check out density reachable or border points from blog to understand how cluster points are considered)
        */
        eps = _eps;
        minimum_points = _min_points;
	}

	torch::Tensor euclidean_distance(torch::Tensor x1, torch::Tensor x2) {
        /*
        :param x1: input tensor
        :param x2: input tensor
        :return: distance between tensors
        */
		if(x1.dim() > 1 && x2.dim() > 1)
			return torch::cdist(x1, x2);
		else
			return torch::norm(x1 - x2, 2);
    }

	torch::Tensor direct_neighbours(int sample) {
        /*
        :param sample: Sample whose neighbors needs to be identified
        :return: all the neighbors within eps distance
        */

		torch::Tensor idxs = torch::arange(X.size(0));
		torch::Tensor sidxs = idxs.masked_select(idxs != sample);
		//std::cout << "sample: " << sample << "\n" << sidxs << '\n';
		torch::Tensor X_slt = X.index_select(0, sidxs);
        std::vector<int> neighbors;

        for(auto& i : range(static_cast<int>(sidxs.size(0)), 0)) {
        	torch::Tensor _sample = X_slt[i];
			torch::Tensor distance = euclidean_distance(X[sample].unsqueeze(0), _sample.unsqueeze(0));
            if( distance.data().item<float>() < eps )
                neighbors.push_back(i);
        }

        return torch::tensor(neighbors);
    }

	std::vector<int> density_neighbors(int sample, torch::Tensor _neighbors) {
        /*
        Recursive method which expands the cluster until we have reached the border
        of the dense area (density determined by eps and min_samples)

        :param sample: Sample whose border points to be identified
        :param neighbors: samples and its neighbors within eps distance
        :return: It updates the number of points assigned to each cluster, by finding
        border points and its relative points. In a sense, it expands cluster.
        */
        std::vector<int> cluster = {sample};

        for( int i = 0; i < _neighbors.size(0); i++ ) {
        	int neighbor_i = _neighbors[i].data().item<int>();
        	int cnt = std::count(visited_samples.begin(), visited_samples.end(), neighbor_i);
            if(cnt < 1) {
                visited_samples.push_back(neighbor_i);
                neighbors[neighbor_i] = direct_neighbours(neighbor_i);
                torch::Tensor d = neighbors[neighbor_i];
                if( d.size(0) >= minimum_points ) {
                	std::vector<int> expanded_cluster = density_neighbors(neighbor_i, d.clone());

                	std::vector<int>  merged;
                    std::merge(cluster.begin(), cluster.end(), expanded_cluster.begin(),
                    		expanded_cluster.end(), merged.begin());
                    cluster = merged;
                } else {
                    cluster.push_back(neighbor_i);
                }
            }
        }
        return cluster;
    }

	torch::Tensor get_cluster_label(void) {
        //:return: assign cluster label based on expanded clusters

		torch::Tensor labels = torch::zeros({X.size(0)}, torch::kInt32).fill_(clusters.size());
        for(auto& cluster_i : range(static_cast<int>(clusters.size()), 0)) {
        	std::vector<int> cluster = clusters[cluster_i];
            for(auto& sample_i : cluster) {
                labels[sample_i] = cluster_i;
            }
        }

        return labels;
    }

	torch::Tensor predict(torch::Tensor _X) {
        /*
        :param X: input tensor
        :return: predicting the labels os samples depending on its distance from clusters
        */
        X = _X.clone();
        clusters.clear();
        visited_samples.clear();
        neighbors.clear();
        int n_samples = _X.size(0);

        for(auto& sample_i : range(n_samples, 0)) {
            //if sample_i in self.visited_samples:
        	if(std::count(visited_samples.begin(), visited_samples.end(), sample_i) > 0)
                continue;
            neighbors[sample_i] = direct_neighbours(sample_i);

            torch::Tensor d = neighbors[sample_i];
            //std::cout << "sample_i: " << sample_i << " " << d.size(0) << '\n';
            if( d.size(0) >= minimum_points ) {
                visited_samples.push_back(sample_i);
                std::vector<int> new_cluster = density_neighbors(sample_i, d);
                clusters.push_back(new_cluster);
            }
        }

        return get_cluster_label();
    }

    double accuracy_score(torch::Tensor y, torch::Tensor y_pred) {
    	c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum( y.squeeze() == y_pred.squeeze(), dim).data().item<int>()*1.0 / y.size(0);
        return accuracy;
    }

private:
	float eps = 0.;
	int minimum_points = 0;
	torch::Tensor X = torch::empty(0);
	std::vector<int> visited_samples = {};
	std::map<int, torch::Tensor> neighbors;
	std::vector<std::vector<int>> clusters = {{}};

};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::vector<int> d = {1, 2, 3};
	int cnt = std::count(d.begin(), d.end(), 0);
	std::cout << cnt << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";

	// # 载入数据
	std::ifstream file;
	std::string path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "iris records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> irisMap;
    irisMap.insert({"Iris-setosa", 0});
    irisMap.insert({"Iris-versicolor", 1});
    irisMap.insert({"Iris-virginica", 2});

    std::cout << "irisMap['Iris-setosa']: " << irisMap["Iris-setosa"] << '\n';
    torch::Tensor X, y;
    // not normalize feature
    std::tie(X, y) = process_data2(file, irisMap, false, false, false);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	X = torch::index_select(X, 0, sidx.squeeze());
	y = torch::index_select(y, 0, sidx.squeeze());

	std::cout << "X size = " << X.sizes() << "\n";
	printVector(tensorTovector(y.to(torch::kDouble)));

    DBScan dbscan(0.15, 5);
    torch::Tensor ypred = dbscan.predict(X);
    printVector(tensorTovector(ypred.to(torch::kDouble)));
    printf("Accuracy Score: %.3f\n", dbscan.accuracy_score(y, ypred));

	std::cout << "Done!\n";
	return 0;
}




