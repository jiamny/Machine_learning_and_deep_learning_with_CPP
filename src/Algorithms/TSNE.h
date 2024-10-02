/*
 * TSNE.h
 *
 *  Created on: Jun 16, 2024
 *      Author: jiamny
 */

#ifndef TSNE_H_
#define TSNE_H_

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <cmath>
#include <random>
#include <limits>

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class TSNE {
public:
	TSNE(int _max_iter=1000,  double _initial_momentum = 0.5, double _final_momentum = 0.8,
			int _eta = 500, double _min_gain = 0.01) {
		max_iter = _max_iter;
		initial_momentum = _initial_momentum;
		final_momentum = _final_momentum;
		eta = _eta;
		min_gain = _min_gain;
	}

	std::tuple<torch::Tensor, torch::Tensor> Hbeta(torch::Tensor D, double beta=1.0) {
	    // Compute the perplexity and the P-row for a specific value of the
	    // precision of a Gaussian distribution.

	    // Compute P-row and corresponding perplexity
		// avoid nan
		if( std::isnan(torch::sum(torch::exp(-1.0*D.clone() * beta)).data().item<double>()))
			D = torch::maximum(D, torch::tensor({1e-12}));

		torch::Tensor P = torch::exp(-1.0*D.clone() * beta);

		// avoid nan
		if( std::isnan((torch::log(torch::sum(P)) + beta * torch::sum(D * P) / torch::sum(P)).data().item<double>()))
			P = torch::maximum(P, torch::tensor({1e-12}));

		torch::Tensor sumP = torch::sum(P);

		torch::Tensor H = torch::log(sumP) + beta * torch::sum(D * P) / sumP;

		// avoid nan
		if( std::isnan(torch::sum(P/sumP).data().item<double>()))
			P = torch::maximum(P, torch::tensor({1e-12}));

		P = P / sumP;

	    return std::make_tuple(H, P);
	}

	torch::Tensor x2p(torch::Tensor X, double tol=1e-5, double perplexity=30.0) {
	    // Performs a binary search to get P-values in such a way that each
	    // conditional Gaussian has the same perplexity.

	    // Initialize some variables
	    std::cout << "Computing pairwise distances...\n";
	    int n = X.size(0), d = X.size(1);
	    c10::OptionalArrayRef<long int> dim = {1};
	    torch::Tensor sum_X = torch::sum(torch::square(X), dim);
	    std::cout << "std::isnan(sum_X): " << std::isnan(torch::sum(sum_X).data().item<double>()) << '\n';
	    torch::Tensor D = torch::add(torch::add(-2 * torch::mm(X, X.t()), sum_X).t(), sum_X);
	    std::cout << "std::isnan(D): " << std::isnan(torch::sum(D).data().item<double>()) << '\n';

	    torch::Tensor P = torch::zeros({n, n}).to(X.dtype()).to(X.device());
	    torch::Tensor beta = torch::ones({n, 1}).to(X.dtype()).to(X.device());
	    torch::Tensor logU = torch::log(torch::tensor({perplexity})).to(X.device());

	    torch::Tensor n_list = torch::arange(n).to(X.device());

	    // Loop over all datapoints
	    for(auto& i : range(n, 0)) {

	        // Print progress
	        if( i % 500 == 0 )
	            printf("Computing P-values for point %d of %d...\n", i, n);

	        // Compute the Gaussian kernel and entropy for the current precision
	        torch::Tensor betamin = torch::empty(0).to(X.device());
	        torch::Tensor betamax = torch::empty(0).to(X.device());
	        torch::Tensor idx;
	        if((i+1) < n)
	        	idx = torch::cat({n_list.index({Slice(0, i)}), n_list.index({Slice(i+1, n)})}, 0);
	        else
	        	idx = n_list.index({Slice(0, i)});

	        torch::Tensor Di = D.index({i, idx});
	        torch::Tensor H, thisP;

	        std::tie(H, thisP) = Hbeta(Di, beta[i].data().item<double>());

	        // Evaluate whether the perplexity is within tolerance
	        torch::Tensor Hdiff = H - logU;
	        int tries = 0;

	        while( torch::abs(Hdiff).data().item<double>() > tol && tries < 50 ) {

	            // If not, increase or decrease precision
	            if( Hdiff.data().item<double>() > 0 ) {

	                betamin = beta[i].clone();
	                if( betamax.numel() == 0 ) {
	                    beta[i] = beta[i] * 2.;
	                } else {
	                    beta[i] = (beta[i] + betamax) / 2.;
	                }
	            } else {
	                betamax = beta[i].clone();
	                if( betamin.numel() == 0 ) {
	                    beta[i] = beta[i] / 2.;
	                } else {
	                    beta[i] = (beta[i] + betamin) / 2.;
	                }
	            }

	            // Recompute the values
	            std::tie(H, thisP) = Hbeta(Di, beta[i].data().item<double>());

	            Hdiff = H - logU;
	            tries += 1;
	        }

	        // Set the final row of P
	        P.index_put_({i, idx}, thisP);
	    }

	    // Return final P-matrix
	    printf("Mean value of sigma: %f\n", torch::mean(torch::sqrt(1 / beta)).data().item<double>());

	    return P;
	}

	torch::Tensor pca_init(torch::Tensor X, int no_dims=50) {
	    // Runs PCA on the NxD array X in order to reduce its dimensionality to
	    // no_dims dimensions.
	    printf("Preprocessing the data using PCA...\n");

	    int n = X.size(0), d = X.size(1);
	    c10::OptionalArrayRef<long int> dim = {0};
	    torch::Tensor t = torch::mean(X, dim);

	    X = X - torch::tile(t, {n, 1});
	    torch::Tensor A = torch::mm(X.t(), X);
	    torch::Tensor l, M;
	    std::tie(l, M) = torch::linalg::eig(A);

	    l = l.to(X.dtype());
	    M = M.to(X.dtype());

	    torch::Tensor Y = torch::mm(X, M.index({Slice(), Slice(0, no_dims)}));

	    return Y;
	}

	torch::Tensor fit_tsne(torch::Tensor X, int no_dims=2, int initial_dims=50, double perplexity=30.0) {
	    // Runs t-SNE on the dataset in the NxD array X to reduce its
	    // dimensionality to no_dims dimensions. The syntaxis of the function is
	    // `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.

	    // Check inputs
	    if(std::round(no_dims) != no_dims ) {
	        printf("Error: number of dimensions should be an integer.\n");
	        exit(-1);
	    }

	    // Initialize variables
	    X = pca_init(X, initial_dims);
	    int n = X.size(0), d = X.size(1);

	    torch::Tensor Y = torch::randn({n, no_dims}).to(X.dtype()).to(X.device());

	    torch::Tensor dY = torch::zeros({n, no_dims}).to(X.dtype()).to(X.device());
	    torch::Tensor iY = torch::zeros({n, no_dims}).to(X.dtype()).to(X.device());
	    torch::Tensor gains = torch::ones({n, no_dims}).to(X.dtype()).to(X.device());

	    // Compute P-values
	    torch::Tensor P = x2p(X, 1e-5, perplexity);
	    P = P + P.t();
		// avoid nan
		if( std::isnan(torch::sum(P).data().item<double>()))
			P = torch::maximum(P, torch::tensor({1e-12}));

	    P = P / torch::sum(P);
	    P = P * 4.;    			// early exaggeration

	    P = torch::maximum(P, torch::tensor({1e-21}));

	    // Run iterations
	    for(auto& iter : range(max_iter, 0)) {
	        // Compute pairwise affinities
	    	c10::OptionalArrayRef<long int> dim = {1};
	    	torch::Tensor sum_Y = torch::sum(torch::square(Y), dim);

	    	torch::Tensor num = -2. * torch::mm(Y, Y.t());
	        num = 1. / (1. + torch::add(torch::add(num, sum_Y).t(), sum_Y));
	        num.index_put_({torch::arange(n), torch::arange(n)}, 0.);

	        torch::Tensor Q = num / torch::sum(num);
	        Q = torch::maximum(Q, torch::tensor({1e-12}));

	        // Compute gradient
	        torch::Tensor PQ = P - Q;

	        for(auto& i : range(n, 0)) {
	            torch::Tensor tmp = PQ.index({Slice(), i}) * num.index({Slice(), i});
	            tmp = torch::tile(tmp, {no_dims, 1}).t();
	            torch::Tensor a = Y.index({i, Slice()}) - Y;

	            at::OptionalIntArrayRef dm = {0};
	        	dY.index_put_({i, Slice()}, torch::sum(tmp * a, dm));
	        }

	        // Perform the update
	        double momentum = 0.;
	        if( iter < 20 )
	            momentum = initial_momentum;
	        else
	            momentum = final_momentum;

	        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).to(X.dtype()) + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).to(X.dtype());
	        gains.masked_fill_(gains < min_gain, min_gain);
	        iY = momentum * iY - eta * (gains * dY);
	        Y = Y + iY;

	        c10::OptionalArrayRef<long int> d = {0};
	        torch::Tensor t = torch::mean(Y, d);
	        Y = Y - torch::tile(t, {n, 1});

	        // Compute current value of cost function
			if( (iter + 1) % 10 == 0) {
				double C = torch::sum(P * torch::log(P / Q)).data().item<double>();
				printf("Iteration %4d / %4d: error is %f\n", (iter + 1), max_iter, C);
			}

	        // Stop lying about P-values
	        if( iter == 100 )
	            P = P / 4.;
	    }

	    // Return solution,
	    return Y;
	}

private:
    int max_iter = 0;
    double initial_momentum = 0.5;
    double final_momentum = 0.8;
    int eta = 500;
    double min_gain = 0.01;
};

#endif /* TSNE_H_ */
