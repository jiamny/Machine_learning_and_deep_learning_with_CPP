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

#include "../Utils/TempHelpFunctions.h"
#include "../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;


class TSNE {
    /*
    The goal is to take a set of points in a high-dimensional space and find a faithful representation of those
    points in a lower-dimensional space, typically the 2D plane. The algorithm is non-linear and adapts to the
    underlying data, performing different transformations on different regions. Those differences can be a major
    source of confusion.
    */
public:
	TSNE(int _n_components=2, double _preplexity=5.0, int _max_iter=1, double _learning_rate=200) {
        /*
        :param n_components:
        :param preplexity: how to balance attention between local and global aspects of your data. The parameter is,
        in a sense, a guess about the number of close neighbors each point has. Typical value between 5 to 50.
        With small value of preplexity, the local groups are formed and with increasing preplexity global groups are
        formed. A perplexity is more or less a target number of neighbors for our central point.
        :param max_iter: Iterations to stabilize the results and converge.
        :param learning_rate:
        */
        max_iter = _max_iter;
        preplexity = _preplexity;
        n_components = _n_components;
        initial_momentum = 0.5;
        final_momentum = 0.8;
        min_gain = 0.01;
        lr = _learning_rate;
        tol = 1e-5;
        preplexity_tries = 50;
	}

    torch::Tensor l2_distance(torch::Tensor  X) {
        //:return: Distance between two vectors
    	c10::OptionalArrayRef<long int> dim = {1};
    	torch::Tensor sum_X = torch::sum(X * X, dim);
    	torch::Tensor T = (-2* torch::mm(X, X.t()) + sum_X).t() + sum_X;

        return T;
    }

    torch::Tensor get_pairwise_affinities(torch::Tensor X) {
        /*
        :param X: High dimensional input
        :return: a (Gaussian) probability distribution over pairs of high-dimensional objects in such a way that similar
        objects are assigned a higher probability while dissimilar points are assigned a lower probability. To find
        variance for this distribution we use Binary search. The variance is calculated between fixed preplexity given
        by the user.
        */
    	torch::Tensor affines = torch::zeros({n_samples, n_samples}, torch::kDouble);
    	torch::Tensor target_entropy = torch::log(torch::tensor(preplexity, torch::kDouble));
    	torch::Tensor distance = l2_distance(X);

        for(auto& i : range(n_samples, 0)) {
        	auto t = binary_search(distance[i], target_entropy);

            affines.index_put_({i, Slice()}, t);
        }

        //printVector(tensorTovector(affines[0].to(torch::kDouble)));

        affines.masked_fill_(torch::eye(affines.size(0)).to(torch::kBool), 1.0e-12);
        std::optional<c10::Scalar> min = 1e-100;
        std::optional<c10::Scalar> max = torch::max(affines).data().item<double>();
        affines = torch::clip(affines, min, max);

        auto t = affines + affines.t();
        affines = t.div(2.0*n_samples);

        return affines;
    }

    torch::Tensor q_distribution(torch::Tensor D) {
        /*
        A (Student t-distirbution)distribution is learnt in lower dimensional space, n_samples and n_components
        (2 or 3 dimension), and similar to above method 'get_pairwise_affinities', we find the probability of the
        data points with high probability for closer points and less probability for disimilar points.
        */
    	torch::Tensor Q = 1.0 / (1.0 + D);
        Q.masked_fill_(torch::eye(Q.size(0)).to(torch::kBool), 0.01);
        std::optional<c10::Scalar> min = 1e-100;
        std::optional<c10::Scalar> max = torch::max(Q).data().item<double>();
        Q = Q.clip(min, max);
        return Q;
    }

    torch::Tensor binary_search(torch::Tensor dist, torch::Tensor target_entropy) {
        /*
        SNE performs a binary search for the value of sigma that produces probability distribution with a fixed
        perplexity that is specified by the user.
        */
        double precision_minimum = 0;
        double precision_maximum = 1.0e15;
        double precision = 1.0e5;
        torch::Tensor beta;

        for(auto& m : range(preplexity_tries, 0)) {
        	torch::Tensor denominator = torch::sum(torch::exp(-dist.masked_select(dist > 0.0) / precision));
            beta = (torch::exp(-dist / precision) / denominator).clone();

            torch::Tensor g_beta = beta.masked_select(beta > 0.0);

			// Shannon Entropy
            torch::Tensor entropy = -torch::sum(g_beta * torch::log2(g_beta));

            torch::Tensor error = entropy - target_entropy;

            if( error.data().item<double>() > 0) {
            	precision_maximum = precision;
            	precision = (precision + precision_minimum) / 2.0;
            } else {
                precision_minimum = precision;
                precision = (precision + precision_maximum) / 2.0;
            }

            if( torch::abs(error).data().item<double>() < tol )
                break;
        }

        return beta;
    }

    torch::Tensor fit_transform(torch::Tensor X) {
        n_samples = X.size(0), n_features = X.size(1);
        torch::Tensor Y = torch::randn({n_samples, n_components}, torch::kDouble);
        /*
        torch::Tensor Y = torch::tensor({{ 1.9269e+00,  1.4873e+00},
            { 9.0072e-01, -2.1055e+00},
            { 6.7842e-01, -1.2345e+00},
            {-4.3067e-02, -1.6047e+00},
            {-7.5214e-01,  1.6487e+00},
            {-3.9248e-01, -1.4036e+00},
            {-7.2788e-01, -5.5943e-01},
            {-7.6884e-01,  7.6245e-01},
            { 1.6423e+00, -1.5960e-01},
            {-4.9740e-01,  4.3959e-01},
            {-7.5813e-01,  1.0783e+00},
            { 8.0080e-01,  1.6806e+00},
            { 1.2791e+00,  1.2964e+00},
            { 6.1047e-01,  1.3347e+00},
            {-2.3162e-01,  4.1759e-02},
            {-2.5158e-01,  8.5986e-01},
            {-1.3847e+00, -8.7124e-01},
            {-2.2337e-01,  1.7174e+00},
            { 3.1888e-01, -4.2452e-01},
            { 3.0572e-01, -7.7459e-01},
            {-1.5576e+00,  9.9564e-01},
            {-8.7979e-01, -6.0114e-01},
            {-1.2742e+00,  2.1228e+00},
            {-1.2347e+00, -4.8791e-01},
            {-9.1382e-01, -6.5814e-01},
            { 7.8024e-02,  5.2581e-01},
            {-4.8799e-01,  1.1914e+00},
            {-8.1401e-01, -7.3599e-01},
            {-1.4032e+00,  3.6004e-02},
            {-6.3477e-02,  6.7561e-01},
            {-9.7807e-02,  1.8446e+00},
            {-1.1845e+00,  1.3835e+00},
            { 1.4451e+00,  8.5641e-01},
            { 2.2181e+00,  5.2317e-01},
            { 3.4665e-01, -1.9733e-01},
            {-1.0546e+00,  1.2780e+00},
            {-1.7219e-01,  5.2379e-01},
            { 5.6622e-02,  4.2630e-01},
            { 5.7501e-01, -6.4172e-01},
            {-2.2064e+00, -7.5080e-01},
            { 1.0868e-02, -3.3874e-01},
            {-1.3407e+00, -5.8537e-01},
            { 5.3619e-01,  5.2462e-01},
            { 1.1412e+00,  5.1644e-02},
            { 7.4395e-01, -4.8158e-01},
            {-1.0495e+00,  6.0390e-01},
            {-1.7223e+00, -8.2777e-01},
            { 1.3347e+00,  4.8354e-01},
            {-2.5095e+00,  4.8800e-01},
            { 7.8459e-01,  2.8647e-02},
            { 6.4076e-01,  5.8325e-01},
            { 1.0669e+00, -4.5015e-01},
            {-1.8527e-01,  7.5276e-01},
            { 4.0476e-01,  1.7847e-01},
            { 2.6491e-01,  1.2732e+00},
            {-1.3109e-03, -3.0360e-01},
            {-1.4570e+00, -1.0234e-01},
            {-5.9915e-01,  4.7706e-01},
            { 7.2618e-01,  9.1152e-02},
            {-3.8907e-01,  5.2792e-01},
            {-1.2685e-02,  2.4084e-01},
            { 1.3254e-01,  7.6424e-01},
            { 1.0950e+00,  3.3989e-01},
            { 7.1997e-01,  4.1141e-01},
            { 1.9312e+00,  1.0119e+00},
            {-1.4364e+00, -1.1299e+00},
            {-1.3603e-01,  1.6354e+00},
            { 6.5474e-01,  5.7600e-01},
            { 1.1415e+00,  1.8565e-02},
            {-1.8058e+00,  9.2543e-01},
            {-3.7534e-01,  1.0331e+00},
            {-6.8665e-01,  6.3681e-01},
            {-9.7267e-01,  9.5846e-01},
            { 1.6192e+00,  1.4506e+00},
            { 2.6948e-01, -2.1038e-01},
            {-7.3280e-01,  1.0430e-01},
            { 3.4875e-01,  9.6759e-01},
            {-4.6569e-01,  1.6048e+00},
            {-2.4801e+00, -4.1754e-01},
            {-1.1955e+00,  8.1234e-01},
            {-1.9006e+00,  2.2858e-01},
            { 2.4859e-02, -3.4595e-01},
            { 2.8683e-01, -7.3084e-01},
            { 1.7482e-01, -1.0939e+00},
            {-1.6022e+00,  1.3529e+00},
            { 1.2888e+00,  5.2295e-02},
            {-1.5469e+00,  7.5671e-01},
            { 7.7552e-01,  2.0265e+00},
            { 3.5818e-02,  1.2059e-01},
            {-8.0566e-01, -2.0758e-01},
            {-9.3195e-01, -1.5910e+00},
            {-1.1360e+00, -5.2260e-01},
            {-5.1877e-01, -1.5013e+00},
            {-1.9267e+00,  1.2785e-01},
            { 1.0229e+00, -5.5580e-01},
            { 7.0427e-01,  7.0988e-01},
            { 1.7744e+00, -9.2155e-01},
            { 9.6245e-01, -3.3702e-01},
            {-1.1753e+00,  3.5806e-01},
            { 4.7877e-01,  1.3537e+00},
            { 5.2606e-01,  2.1120e+00},
            {-5.2076e-01, -9.3201e-01},
            { 1.8516e-01,  1.0687e+00},
            { 1.3065e+00,  4.5983e-01},
            {-8.1463e-01, -1.0212e+00},
            {-4.9492e-01, -5.9225e-01},
            { 1.5432e-01,  4.4077e-01},
            {-1.4829e-01, -2.3184e+00},
            {-3.9800e-01,  1.0805e+00},
            {-1.7809e+00,  1.5080e+00},
            { 3.0943e-01, -5.0031e-01},
            { 1.0350e+00,  1.6896e+00},
            {-4.5051e-03,  1.6668e+00},
            { 1.5392e-01, -1.0603e+00},
            {-5.7266e-01,  8.3568e-02},
            { 3.9991e-01,  1.9892e+00},
            {-7.1988e-02, -9.0609e-01},
            {-2.0487e+00, -1.0811e+00},
            { 1.7623e-02,  7.8226e-02},
            { 1.9316e-01,  4.0967e-01},
            {-9.2913e-01,  2.7619e-01},
            {-5.3888e-01,  4.6258e-01},
            {-8.7189e-01, -2.7118e-02},
            {-3.5325e-01,  1.4639e+00},
            { 1.2554e+00, -7.1496e-01},
            { 8.5392e-01,  5.1299e-01},
            { 5.3973e-01,  5.6551e-01},
            { 5.0579e-01,  2.2245e-01},
            {-6.8548e-01,  5.6356e-01},
            {-1.5072e+00, -1.6107e+00},
            {-1.4790e+00,  4.3227e-01},
            {-1.2503e-01,  7.8212e-01},
            {-1.5988e+00, -1.0913e-01},
            { 7.1520e-01,  3.9139e-02},
            { 1.3059e+00,  2.4659e-01},
            {-1.9776e+00,  1.7896e-02},
            {-1.3793e+00,  6.2580e-01},
            {-2.5850e+00, -2.4000e-02},
            {-1.2219e-01, -7.4700e-01},
            { 1.7093e+00,  5.7923e-02},
            { 1.1930e+00,  1.9373e+00},
            { 7.2871e-01,  9.8089e-01},
            { 4.1459e-01,  1.1566e+00},
            { 2.6905e-01, -3.6629e-02},
            { 9.7329e-01, -1.0151e+00},
            {-5.4192e-01, -4.4102e-01},
            {-3.1362e-01, -1.2925e-01},
            {-7.1496e-01, -4.7562e-02},
            { 2.0207e+00,  2.5392e-01},
            { 9.3644e-01,  7.1224e-01},
            {-3.1766e-02,  1.0164e-01},
            { 1.3433e+00,  7.1327e-01},
            { 4.0380e-01, -7.1398e-01},
            { 8.3373e-01, -9.5855e-01},
            { 4.5363e-01,  1.2461e+00},
            {-2.3065e+00, -1.2869e+00},
            { 1.7989e-01, -2.1268e+00},
            {-1.3408e-01, -1.0408e+00},
            {-7.6472e-01, -5.5283e-02},
            { 1.2049e+00, -9.8247e-01},
            { 4.3344e-01, -7.1719e-01},
            { 1.0554e+00, -1.4534e+00},
            { 4.6515e-01,  3.7139e-01},
            {-4.6568e-03,  7.9549e-02},
            { 3.7818e-01,  7.0511e-01},
            {-1.7237e+00, -8.4348e-01},
            { 4.3514e-01,  2.6589e-01},
            {-5.8710e-01,  8.2689e-02},
            { 8.8538e-01,  1.8244e-01},
            { 7.8638e-01, -5.7920e-02},
            { 5.6667e-01, -7.0976e-01},
            {-4.8751e-01,  5.0096e-02},
            { 6.0841e-01,  1.6309e+00},
            {-8.4723e-02,  1.0844e+00},
            { 9.4777e-01, -6.7663e-01},
            {-5.7302e-01, -3.3032e-01},
            {-7.9394e-01,  3.7523e-01},
            { 8.7910e-02, -1.2415e+00},
            {-3.2025e-01, -8.4438e-01},
            {-5.5135e-01,  1.9890e+00},
            { 1.9003e+00,  1.6951e+00},
            { 2.8090e-02, -1.7537e-01},
            {-1.7735e+00, -7.0464e-01},
            {-3.9465e-01,  1.8868e+00},
            {-2.1844e-01,  1.6630e-01},
            { 2.1442e+00,  1.7046e+00},
            { 3.4590e-01,  6.4248e-01},
            {-2.0395e-01,  6.8537e-01},
            {-1.3969e-01, -1.1808e+00},
            {-1.2829e+00,  4.4849e-01},
            {-5.9074e-01,  8.5406e-01},
            {-4.9007e-01, -3.5946e-01},
            { 6.6637e-01, -7.4265e-02},
            {-2.0960e-01,  1.6632e-01},
            { 1.4703e+00, -9.3909e-01},
            {-6.0132e-01, -9.9640e-02},
            {-9.8515e-01, -2.4885e+00},
            {-3.3132e-01,  8.4358e-01},
            { 9.8745e-01, -3.3197e-01},
            {-8.0762e-01,  8.2436e-01},
            { 2.4700e-02, -1.0641e+00},
            {-7.6019e-01, -4.0751e-01},
            { 9.6236e-01, -1.4264e-01},
            { 1.5271e-01, -3.8802e-02},
            { 9.4461e-01, -1.5824e+00},
            { 9.8713e-01,  1.1457e+00},
            {-1.4181e-01, -2.7634e-01},
            {-1.9321e-01,  7.7678e-01},
            { 6.8388e-01, -1.3246e+00},
            {-5.1608e-01,  6.0018e-01},
            {-4.7022e-01, -6.0864e-01},
            {-4.6192e-02, -1.6457e+00},
            {-4.8333e-01, -7.4029e-01},
            { 3.1428e-01,  1.4156e-01},
            { 1.0348e+00, -6.2644e-01},
            {-5.1509e-01,  6.9029e-01},
            {-4.9400e-01,  1.1366e+00},
            {-4.6184e-01,  1.4200e+00},
            { 8.4852e-01, -4.7891e-02},
            { 6.6856e-01,  1.0430e+00},
            { 6.8990e-01, -1.3129e+00},
            { 3.7804e-02, -1.1702e+00},
            {-1.0319e-01,  1.1895e+00},
            { 7.6069e-01, -7.4630e-01},
            {-1.3839e+00,  4.8687e-01},
            {-1.0020e+00,  3.2949e-02},
            {-4.2920e-01, -9.8180e-01},
            {-6.4206e-01,  8.2659e-01},
            { 1.5914e+00, -1.2081e-01},
            {-4.8302e-01,  1.1330e-01},
            { 7.7151e-02, -9.2281e-01},
            {-1.2620e+00,  1.0861e+00},
            { 1.0966e+00, -6.8369e-01},
            { 6.6043e-02, -7.7380e-04},
            { 1.6206e-01,  1.1960e+00},
            {-1.3062e+00, -1.4040e+00},
            {-1.0597e+00,  3.0573e-01},
            { 4.1506e-01, -7.1741e-01},
            { 2.8340e+00,  1.9535e+00},
            { 2.0487e+00, -1.0880e+00},
            { 1.6217e+00,  8.5127e-01},
            {-4.0047e-01, -6.0883e-01},
            {-5.0810e-01, -6.1849e-01},
            {-1.6470e+00, -1.0362e+00},
            {-4.5031e-01, -7.2966e-02},
            {-5.4795e-01, -1.1426e+00},
            {-4.4875e-01, -3.0454e-02},
            { 3.8303e-01, -4.4770e-02},
            { 1.1799e+00, -3.3143e-01},
            { 6.4950e-01,  9.4959e-02},
            {-7.5259e-01, -6.4723e-01},
            {-1.2823e+00,  1.9653e+00},
            {-9.6385e-01, -2.5668e+00},
            { 7.0961e-01,  8.1984e-01},
            { 6.2145e-01,  4.2319e-01},
            {-3.3890e-01,  5.1797e-01},
            {-1.3638e+00,  1.9296e-01},
            {-6.1033e-01,  1.6323e-01},
            { 1.5102e+00,  2.1230e-01},
            {-7.2520e-01, -9.5277e-01},
            { 5.2169e-01, -4.6387e-01},
            { 1.8238e-01, -3.8666e-01},
            {-1.7907e+00,  9.3293e-02},
            {-1.9153e+00, -6.4218e-01},
            { 1.3439e+00, -1.2922e+00},
            { 7.6624e-01,  6.4540e-01},
            { 3.5332e-01, -2.6475e+00},
            {-1.4575e+00, -9.7124e-01},
            { 2.5403e-01, -1.7906e-01},
            { 1.1993e+00, -4.2922e-01},
            { 1.0103e+00,  6.1104e-01},
            { 1.2208e+00, -6.0764e-01},
            {-1.7376e+00, -1.2535e-01},
            {-1.3658e+00,  1.1117e+00},
            {-6.2280e-01, -7.8918e-01},
            {-1.6782e-01,  1.6433e+00},
            { 2.0071e+00, -1.2531e+00},
            { 1.1189e+00,  1.7733e+00},
            {-2.0717e+00, -4.1253e-01},
            {-9.7696e-01, -3.3634e-02},
            { 1.8595e+00,  2.6221e+00},
            { 3.6905e-01,  3.8030e-01},
            { 1.9898e-01, -2.3609e-01},
            { 3.0341e-01, -4.5008e-01},
            { 4.7390e-01,  6.5034e-01},
            { 1.1662e+00,  1.6936e-02},
            { 5.3259e-01, -6.0354e-01},
            {-1.7426e-01,  6.0921e-01},
            {-8.0322e-01, -1.1209e+00},
            { 1.9564e-01, -7.8152e-01},
            {-1.7899e+00, -2.6157e-01},
            {-4.4025e-01,  2.1848e+00},
            {-4.8010e-01, -1.2872e+00},
            { 7.3888e-01,  3.3895e-02},
            {-3.1229e-01, -2.5418e-01},
            {-1.2055e+00, -9.5421e-01},
            { 6.1277e-02,  8.5261e-02},
            { 7.4813e-01, -1.6356e-01},
            {-9.0856e-01,  3.1300e-01},
            { 8.0505e-01, -1.1134e+00},
            { 4.9816e-01, -1.2000e+00},
            { 1.2711e-01,  4.4037e-01},
            { 6.3777e-01,  1.5979e-01},
            { 1.7698e+00,  6.2682e-01},
            {-1.8737e+00,  2.3259e+00},
            {-9.2039e-01,  6.6611e-01},
            {-4.4026e-01, -2.3180e+00},
            { 1.2946e+00,  2.2267e-01},
            {-8.4834e-01,  1.6489e+00},
            { 1.6006e+00, -7.8589e-02},
            { 4.3105e-01,  3.6835e-01},
            { 7.6380e-01,  1.1792e+00},
            {-4.1379e-01,  5.1841e-01},
            {-7.0154e-01, -4.3234e-01},
            { 1.4148e-01,  7.1104e-02},
            { 5.6335e-01, -5.7864e-01},
            {-1.0838e+00, -3.8893e-01},
            { 8.1261e-01,  1.4981e+00},
            { 4.3896e-02,  1.4443e+00},
            { 2.3203e-01,  5.0650e-01},
            {-1.2787e+00, -3.8427e-02},
            { 1.9138e+00,  3.3784e-01},
            { 1.2506e-01, -7.6215e-01},
            {-1.1906e+00,  7.7561e-01},
            { 4.5572e-01,  2.5033e-01},
            {-1.3611e+00,  1.8018e+00},
            {-7.4342e-02, -1.5664e-01},
            {-8.7085e-01, -6.4110e-01},
            {-4.1456e-01, -6.9024e-01},
            {-2.2996e-01, -2.1723e+00},
            { 8.7683e-02,  1.0938e+00},
            {-1.1772e-01, -2.9864e-01},
            {-9.5362e-01, -9.2473e-02},
            {-1.0167e+00, -7.6757e-03},
            {-5.1822e-01,  8.3954e-01},
            { 5.8523e-02, -1.6682e+00},
            { 2.1296e+00, -1.5181e+00},
            { 1.3873e-01, -1.1798e+00},
            {-5.2974e-01,  9.6252e-01},
            { 2.7944e-01, -5.7182e-01},
            {-2.7936e+00, -7.1115e-01},
            { 5.2352e-01, -1.7106e+00},
            { 8.3849e-01, -2.6985e-01},
            { 1.2306e-01,  8.7575e-01},
            { 1.5133e-01,  7.3939e-01},
            { 2.7310e-01,  2.7312e+00},
            { 4.3201e-01, -3.0918e-01},
            {-9.6581e-02,  1.5419e+00},
            {-1.0874e-01, -4.1890e-01},
            { 1.4384e+00, -7.0684e-01},
            {-1.2520e+00,  3.0250e+00},
            { 1.3463e+00,  8.5561e-01},
            { 3.2203e-01,  4.4606e-01},
            { 1.5230e+00,  1.2805e+00},
            {-1.1616e-01,  1.3705e+00},
            {-4.8094e-01, -9.9036e-01},
            {-1.3642e+00,  8.2057e-03},
            {-4.0586e-01, -7.1109e-01},
            {-3.4958e-01,  3.7975e-01},
            { 9.9930e-01,  1.2752e+00},
            { 9.5949e-01,  1.0351e-01},
            { 8.2903e-01,  2.0921e+00},
            { 7.9531e-01,  2.7928e-01},
            { 1.8645e-01,  3.5471e-01},
            { 9.0639e-02,  1.7423e+00},
            {-1.2660e+00,  3.8916e-01},
            { 3.4288e-01, -1.4591e+00},
            {-1.4937e+00, -2.2139e-01},
            { 2.2524e-01, -7.7245e-02},
            { 9.8569e-01,  1.2783e+00},
            { 2.8815e-01,  8.6905e-01},
            {-8.0971e-01, -1.4299e+00},
            { 4.5902e-01,  5.3093e-01},
            {-1.3615e+00,  1.9562e+00},
            { 1.7685e+00, -9.8580e-01},
            {-1.2371e+00, -2.3019e+00},
            {-1.0087e-03, -8.4943e-01},
            {-1.6594e+00,  3.0629e-01},
            { 1.1820e+00,  3.2603e-01},
            {-3.8945e-01,  2.8544e+00},
            { 8.2437e-01,  7.9835e-01},
            { 1.8890e+00,  5.9346e-01},
            { 6.9654e-02, -1.6034e+00},
            {-4.2982e-01,  5.7616e-01},
            { 3.4436e-01, -3.1016e+00},
            {-1.4587e+00, -1.4318e+00},
            {-6.0713e-01, -2.5974e-01},
            {-7.1902e-01, -3.8583e-01},
            { 5.2335e-01, -8.2118e-01},
            {-4.7087e-01,  6.0164e-01},
            {-2.8251e-01,  7.6927e-01},
            {-7.6689e-01, -9.4949e-01},
            { 1.6917e-02,  8.0277e-02},
            { 7.4484e-01,  1.3455e+00},
            { 1.2682e-01, -2.4521e+00},
            { 4.1598e-01,  1.9025e+00},
            {-7.3467e-01,  4.4657e-02},
            {-1.5211e+00,  3.4784e-01},
            { 7.4018e-01,  1.4162e+00},
            { 6.8340e-01, -1.3825e-01},
            { 9.2130e-01,  5.2824e-01},
            {-8.2284e-03, -1.4493e+00},
            {-6.0518e-01, -1.7925e-01},
            { 1.9956e-01, -1.2462e+00},
            {-4.1460e-01,  1.4559e+00},
            { 3.3165e-01, -1.0001e+00},
            {-6.9195e-01, -4.7199e-01},
            {-1.2894e+00,  1.0763e+00},
            {-1.0667e+00, -1.9893e+00},
            { 2.9731e-01,  4.3446e-01},
            { 3.3933e-03, -1.0240e+00},
            { 2.2405e-01, -7.5548e-01},
            { 1.3676e+00, -3.1974e-01},
            {-9.1309e-01,  1.9192e+00},
            {-1.6515e+00,  2.1477e+00},
            {-6.6041e-01,  1.1353e-01},
            {-2.2057e-01,  7.1181e-01},
            { 3.4159e-01,  1.5886e+00},
            {-3.4888e-01, -4.5792e-01},
            {-1.2322e+00, -5.9808e-01},
            {-2.8155e-01,  5.2819e-02},
            { 4.2498e-01,  4.8258e-01},
            { 4.8813e-01,  1.0082e+00},
            {-5.9500e-01,  3.9263e-01},
            { 8.2297e-01, -8.8603e-01},
            { 1.4801e+00,  8.3915e-01},
            {-2.0005e-01,  9.9495e-01},
            { 7.2019e-01, -1.3413e-01},
            {-1.4068e+00, -2.3610e+00},
            {-2.9049e-01, -1.3346e-01},
            {-1.5693e-01,  1.1383e+00},
            {-2.5052e-01,  1.6705e+00},
            {-5.4527e-01, -2.1582e+00},
            {-1.6608e+00, -6.6374e-01},
            {-1.5844e+00, -2.5447e+00},
            { 1.3719e+00, -5.3795e-01},
            {-1.7564e-01, -3.6833e-01},
            {-8.3345e-01, -3.3530e-01},
            { 9.2000e-01, -3.7876e-01},
            {-1.5598e+00, -8.0095e-01},
            { 3.5879e-01,  1.2862e+00},
            { 8.2114e-01,  9.0019e-01}}, torch::kDouble);
		*/
        torch::Tensor velocity = torch::zeros_like(Y);
        torch::Tensor gains = torch::ones_like(Y);
        torch::Tensor P = get_pairwise_affinities(X);
        //std::cout << "P[0]:\n";
        //printVector(tensorTovector(P[0]));

        int iter_num = 0;
        int one_step = 1;
        if( max_iter > 10 )
            one_step = static_cast<int>(max_iter / 10.0);

        while( iter_num < max_iter ) {
            iter_num += 1;
            torch::Tensor D = l2_distance(Y);
            //std::cout << "D[0]:\n";
            //printVector(tensorTovector(D[0]));
            torch::Tensor Q = q_distribution(D);
            torch::Tensor Q_n = Q /torch::sum(Q);

            double pmul = 4.0;
            if( iter_num >= 100 )
            	pmul = 1.0;
            double momentum = 0.5;
            if( iter_num >= 20 )
            	momentum = 0.8;

            torch::Tensor grads = torch::zeros(Y.sizes(), torch::kDouble);
            for(auto& i : range(n_samples, 0)) {
                //Optimization using gradient to converge between the true P and estimated Q distrbution.
            	torch::Tensor grad = 4 * torch::mm(((pmul * P[i] - Q_n[i]) * Q[i]).unsqueeze(0), Y[i] -Y);
                grads.index_put_({i, Slice()}, grad);
            }

            gains = (gains + 0.2) * ((grads > 0) != (velocity > 0)).to(torch::kInt32) +
            		(gains * 0.8) * ((grads > 0) == (velocity > 0)).to(torch::kInt32);

        	std::optional<c10::Scalar> min = min_gain;
        	std::optional<c10::Scalar> max = torch::max(gains).data().item<double>();
            gains = gains.clip(min, max);

            velocity = momentum * velocity - lr * (gains * grads);

            Y += velocity;
            c10::OptionalArrayRef<long int> dim = {0};
            Y = Y - torch::mean(Y, dim);

            torch::Tensor error = torch::sum(P * torch::log(P/Q_n));

            if( iter_num % one_step == 0  || iter_num == max_iter ) {
            	std::cout << "Iteration: " << iter_num << ", error: " <<  error.data().item<double>() << '\n';
            }
        }
        return Y;
    }

private:
        int max_iter = 0, n_components = 0, preplexity_tries = 50, n_samples = 0, n_features = 0;
        double preplexity = 0., initial_momentum = 0.5, final_momentum = 0.8, min_gain = 0.01;
        double lr = 0.1, tol = 1e-5;
};

#endif /* TSNE_H_ */
