/*
 * EM_based_PLSA.cpp
 *
 *  Created on: May 31, 2024
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
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

class PLSA {
public:
	PLSA( int _K, int _max_iter) {
        K = _K;
        max_iter = _max_iter;
	}

	std::pair<torch::Tensor, torch::Tensor> fit(torch::Tensor X) {
        int n_d = X.size(0), n_w = X.size(1);

        // P(z|w,d)
        torch::Tensor p_z_dw = torch::zeros({n_d, n_w, K});

        // P(z|d)
        torch::Tensor p_z_d = torch::rand({n_d, K});

        // P(w|z)
        torch::Tensor p_w_z = torch::rand({K, n_w});

        for(auto& i_iter : range(max_iter, 0) ) {
            // E step
            for(auto& di : range(n_d, 0)) {
                for(auto& wi : range(n_w, 0)) {
                	torch::Tensor sum_zk = torch::zeros({K});
                    for(auto& zi : range(K, 0))
                        sum_zk[zi] = p_z_d[di][zi] * p_w_z[zi][wi];
                    double sum1 = torch::sum(sum_zk).data().item<double>();
                    if( sum1 == 0 )
                        sum1 = 1;
                    for(auto& zi : range(K, 0))
                        p_z_dw.index_put_({di, wi, zi}, sum_zk[zi] / sum1);
                }
            }
            // M step

            // update P(z|d)
            for(auto& di : range(n_d, 0)) {
                for(auto& zi : range(K, 0)) {
                    double sum1 = 0.;
                    double sum2 = 0.;

                    for(auto& wi : range(n_w, 0)) {
                        sum1 = sum1 + X[di][wi].data().item<double>() * p_z_dw[di][wi][zi].data().item<double>();
                        sum2 = sum2 + X[di][wi].data().item<double>();
                    }

                    if(sum2 == 0)
                        sum2 = 1;
                    p_z_d[di][zi] = sum1 / sum2;
                }
            }

            // update P(w|z)
            for(auto& zi : range(K, 0)) {
            	torch::Tensor sum2 = torch::zeros({n_w});
                for(auto& wi : range(n_w, 0)) {
                    for(auto& di : range(n_d, 0)) {
                        sum2[wi] = sum2[wi] + X[di][wi] * p_z_dw[di][wi][zi];
                    }
                }
                double sum1 = torch::sum(sum2).data().item<double>();
                if(sum1 == 0 ) {
                    sum1 = 1;
                    for(auto& wi : range(n_w, 0)) {
                        p_w_z[zi][wi] = sum2[wi] / sum1;
                    }
                }
            }
        }

        return std::make_pair(p_w_z, p_z_d);
	}

private:
	int K = 2, max_iter = 100;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Tensor X = torch::tensor(
		{{0,0,1,1,0,0,0,0,0},
	     {0,0,0,0,0,1,0,0,1},
	     {0,1,0,0,0,0,0,1,0},
	     {0,0,0,0,0,0,1,0,1},
	     {1,0,0,0,0,1,0,0,0},
	     {1,1,1,1,1,1,1,1,1},
	     {1,0,1,0,0,0,0,0,0},
	     {0,0,0,0,0,0,1,0,1},
	     {0,0,0,0,0,2,0,0,1},
	     {1,0,1,0,0,0,0,1,0},
	     {0,0,0,1,1,0,0,0,0}});

	/*
	X[0][2] = 5;
	std::cout << X.sizes() << '\n' << X[0][2] << '\n';
	*/
	X = X.t();
	auto model = PLSA(2, 100);

	torch::Tensor p_w_z, p_z_d;
	std::tie( p_w_z, p_z_d ) = model.fit(X);

	std::cout << "p_w_z:\n" << p_w_z << '\n';
	std::cout << "p_z_d:\n" << p_z_d << '\n';

	std::cout << "Done!\n";
	return 0;
}






