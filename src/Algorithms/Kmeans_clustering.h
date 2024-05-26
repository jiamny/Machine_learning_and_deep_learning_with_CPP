/*
 * Kmeans_clustering.hpp
 *
 *  Created on: May 4, 2024
 *      Author: jiamny
 */

#ifndef KMEANS_CLUSTERING_HPP_
#define KMEANS_CLUSTERING_HPP_
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>
#include <float.h>
#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class KMeans {
public:
    int64_t k=0, samples=0, features=0, max_iterations=0;
    torch::Tensor KMeans_Centroids;
    std::string method = "";
    /**
     @param X: input tensor
     @param k: Number of clusters
     :variable samples: Number of samples
     :variable features: Number of features
     */
    KMeans(torch::Tensor  X, int64_t _k, int64_t iterations, std::string _method) {
        k = _k;
        max_iterations = iterations;
        samples = X.size(0);
        features = X.size(1);
        KMeans_Centroids = torch::empty(0);
        method = _method;
    }

    /*
    Initialization Technique is KMeans++. Thanks to stackoverflow.
        https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        :param X: Input Tensor
        :param K: Number of clusters to build
        :return: Selection of three centroid vector from X
     */

    torch::Tensor initialize_centroid(torch::Tensor X, int64_t K) {

        std::vector<int64_t> I;
        std::vector<torch::Tensor> C;

        I.push_back(0);
        C.push_back( X.index({0, Slice()}) );

        for( int64_t k = 1; k < K; k++ ) {
        	torch::Tensor D2 = torch::zeros({samples}).to(X.dtype());

            for (int64_t i = 0; i < samples; i++) {
            	torch::Tensor x = X.index({i, Slice()});
            	torch::Tensor cd = torch::zeros({static_cast<int64_t>(C.size())}).to(X.dtype());
                //D2 = np.array([min([np.inner(c - x, c - x) for c in C]) for x in X])
                for(int64_t j = 0; j < C.size(); j++) {
                	torch::Tensor c = C[j];
                	torch::Tensor t = c.sub(x);
                    cd.index_put_(c10::List<std::optional<at::Tensor>>{torch::tensor({j})}, t.dot(t));
                }
                D2.index_put_(c10::List<std::optional<at::Tensor>>{torch::tensor({i})}, cd.min());
            }
            torch::Tensor probs = D2.div(D2.sum());
            torch::Tensor cumprobs = torch::cumsum(probs, 0);
            //std::cout << "cumprobs: " << cumprobs.sizes() <<'\n';
            double r = torch::rand({1}, torch::kDouble).data().item<double>();;

            //r = torch.rand(1).item()
            int64_t l = cumprobs.size(0);
            int64_t idx = 0;
            for(int64_t n = 0; n < l; n++) {
                if( r < cumprobs[n].data().item<double>() ) {
                    idx = n;
                    break;
                }
            }
            I.push_back(idx);
        }
        torch::Tensor sidx = torch::from_blob(I.data(),
    			{static_cast<long int>(I.size())}, at::TensorOptions(torch::kLong)).clone();
        //std::cout << X.index({sidx, Slice()}) <<'\n';
        return X.index({sidx, Slice()});
    }

    torch::Tensor distance(torch::Tensor sample, torch::Tensor centroid ) {
        if( method == "euclidean" ){
            return torch::norm( sample.sub(centroid), 2); //.norm(2.0); //torch.norm(sample - centroid, 2, 0)
        } else if( method == "manhattan" ){
            //return torch.sum(torch.abs(sample - centroid))
            return (sample.sub(centroid).abs()).sum();
        } else if( method == "cosine" ) {
            //std::cout << (torch::norm(sample) * torch::norm(centroid)) <<'\n';
            return torch::sum(sample * centroid) / (torch::norm(sample) * torch::norm(centroid));
        } else {
                std::cout <<"Unknown similarity distance type\n";
                return torch::empty(0);
        }
    }

    /*
    :param sample: sample whose distance from centroid is to be measured
    :param centroids: all the centroids of all the clusters
    :return: centroid's index is passed for each sample
     */
    int64_t closest_centroid(torch::Tensor sample, torch::Tensor centroids) {
        int closest = -1;
        double min_distance = DBL_MAX;
        for( int64_t idx = 0; idx < centroids.size(0); idx++ ) {
        	torch::Tensor centroid = centroids.index({idx, Slice()});
            torch::Tensor d = distance(sample, centroid);
            if( d.numel() > 0 ) {
            	double distance = d.data().item<double>();
            	if (distance < min_distance) {
            		closest = idx;
            		min_distance = distance;
            	}
            }
        }
        return closest;
    }

    /*
     :param centroids: Centroids of all clusters
     :param X: Input tensor
     :return: Assigning each sample to a cluster.
     */
    std::vector<std::vector<int64_t>> create_clusters(torch::Tensor centroids, torch::Tensor X) {
        int64_t n_samples = X.size(0);
        std::vector<std::vector<int64_t>> k_clusters; //[[] for _ in range(self.k)]

        for(int64_t j = 0; j < k; j++) {
        	std::vector<int64_t> kc;
            k_clusters.push_back(kc);
        }

        for(int64_t idx = 0; idx < n_samples; idx++) {
        	torch::Tensor sample = X.index({idx, Slice()});
            int64_t centroid_index = closest_centroid(sample, centroids);
            k_clusters[centroid_index].push_back(idx);
        }

        return k_clusters;
    }

    /*
    :return: Updating centroids after each iteration.
     */
    torch::Tensor update_centroids(std::vector<std::vector<int64_t>> clusters, torch::Tensor X) {

    	torch::Tensor centroids = torch::zeros({k, features}, X.dtype() );
        for( int64_t idx = 0; idx < k; idx++) {
            //NDArray cIdx = manager.create(clusters.get(idx).stream().mapToInt(m -> m).toArray()).toType(DataType.INT32, false);
        	torch::Tensor cidx = torch::from_blob(clusters[idx].data(),
        			{static_cast<long int>(clusters[idx].size())}, at::TensorOptions(torch::kLong)).clone();
        	torch::Tensor centroid = torch::index_select(X, 0, cidx.squeeze());;

        	c10::OptionalArrayRef<long int> didx = {0};
            torch::Tensor mn = centroid.mean(didx);

            centroids.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({idx})}, mn);
        }
        return centroids;
    }

    /*
    Labeling the samples with index of clusters
    :return: labeled samples
     */
    torch::Tensor label_clusters( std::vector<std::vector<int64_t>> clusters, torch::Tensor X) {

    	torch::Tensor y_pred = torch::zeros({X.size(0)}, torch::kLong);

        for(int64_t idx = 0; idx < k; idx++) {
        	std::vector<int64_t> cluster = clusters[idx];
        	//printVector(cluster);

            for( int64_t sample_idx : cluster ) // y_pred.set(new NDIndex(sample_idx), idx);
            	y_pred.index_put_(c10::ArrayRef<at::indexing::TensorIndex>{torch::tensor({sample_idx})}, idx);
        }
        return y_pred;
    }

    /*
        Initializing centroid using Kmeans++, then find distance between each sample and initial centroids, then assign
        cluster label based on min_distance, repeat this process for max_iteration and simultaneously updating
        centroid by calculating distance between sample and updated centroid. Convergence happen when difference between
        previous and updated centroid is None.
     */
    void fit(torch::Tensor X) {
    	torch::Tensor centroids = initialize_centroid(X, k);
        for( int64_t i = 0; i < max_iterations; i++ ) {
        	std::vector<std::vector<int64_t>> clusters = create_clusters(centroids, X);
        	torch::Tensor previous_centroids = centroids.clone();

            centroids = update_centroids(clusters, X);

            torch::Tensor difference = centroids.sub(previous_centroids);

            if( difference.sum().data().item<double>() != 0.0 )
                continue;
            else
                break;
        }
        KMeans_Centroids = centroids;
    }

    /*
    :return: label/cluster number for each input sample is returned
     */
    torch::Tensor predict(torch::Tensor X) {
        if( KMeans_Centroids.numel() < 1 ) {
            std::cout << "No Centroids Found. Run KMeans fit\n";
            return torch::empty(0);
        }

        std::vector<std::vector<int64_t>> clusters = create_clusters(KMeans_Centroids, X);
        torch::Tensor labels = label_clusters(clusters, X);

        return labels;
    }

    double score(torch::Tensor X_ts, torch::Tensor y_ts, bool verbose = false) {
        torch::Tensor y_pred = predict(X_ts).squeeze_();
        if( verbose ) {
        	std::cout << "y_ts: " << y_ts.sizes() << '\n';
        	std::cout << "y_pred: " << y_pred.sizes() << '\n';
        	printVector(tensorTovector(y_pred.to(torch::kDouble)));
        }
        c10::OptionalArrayRef<long int> dim = {0};
        double accuracy = torch::sum(y_ts.squeeze() == y_pred, dim).data().item<double>() / X_ts.size(0);
        return accuracy;
    }
};


#endif /* KMEANS_CLUSTERING_HPP_ */
