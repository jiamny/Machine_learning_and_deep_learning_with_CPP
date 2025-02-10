/*
 * Probabilstic_PCA.cpp
 *
 *  Created on: Aug 24, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// updating W, X and tau when doing inference

class PPCA {
public:
    // Y: input continuous data with shape (N, M)
    // D: number of ppca components
	PPCA(int _D=2, int _n_iters=100, bool _verbose=false) {
        D = _D;
        n_iters = _n_iters;
        verbose = _verbose;
	}

    void _init_paras(int N, int M, int D) {
        a = 1.0;
        b = 1.0;
        e_tau = a / b;
        e_w = torch::randn({M, D});
        e_wwt = torch::zeros({D, D, M}).to(torch::kFloat32);
        for(auto& m : range(M, 0)) {
            // use np.newaxis here to transfer numpy array from 1D to 2D
        	torch::Tensor tt = e_w.index({m, Slice()}).unsqueeze(0);
        	torch::Tensor dt = torch::eye(D).to(torch::kFloat32) + torch::mm(tt.t(), tt);
        			//[m, :][np.newaxis].T.dot(self.e_w[m, :][np.newaxis]);
            e_wwt.index_put_({Slice(), Slice(), m}, dt); //[:, :, m] =
        }

        tol = 1e-3;
        lbs.clear();
        e_X = torch::zeros({N, D}).to(torch::kFloat32);
        e_XXt = torch::zeros({D, D, N}).to(torch::kFloat32);
    }

    void _update_X(torch::Tensor Y, int N, int D) {
    	c10::OptionalArrayRef<long int> dim = {2};
        sigx = torch::linalg_inv(torch::eye(D).to(torch::kFloat32) + e_tau * torch::sum(e_wwt, dim));

        for(auto& n : range(N, 0)) {
        	c10::OptionalArrayRef<long int> axis = {0};
        	std::optional<long int> dm = {0};
        	torch::Tensor rt = torch::repeat_interleave(Y[n].unsqueeze(0), D, dm).t();
        	torch::Tensor d = e_tau * sigx.mm(torch::sum(e_w * rt, axis).reshape({-1, 1}));
            e_X.index_put_({n, Slice()}, d.squeeze());

            torch::Tensor dt = e_X[n].unsqueeze(0);
            e_XXt.index_put_({Slice(), Slice(), n}, sigx + torch::mm(dt.t(), dt));
        }
    }

    void _update_W(torch::Tensor Y, int M, int D) {
    	c10::OptionalArrayRef<long int> axis = {2};
        sigw = torch::linalg_inv(torch::eye(D).to(torch::kFloat32) + e_tau * torch::sum(e_XXt, axis));

        for(auto& m : range(M, 0)) {
        	c10::OptionalArrayRef<long int> dim = {0};
        	std::optional<long int> dm = {0};
        	torch::Tensor rt = torch::repeat_interleave(Y.index({Slice(), m}).unsqueeze(0), 2, dm).t();
        	torch::Tensor d = e_tau * sigw.mm(torch::sum(e_X * rt, dim).reshape({-1, 1}));
            e_w.index_put_({m, Slice()}, d.squeeze());

            torch::Tensor dt = e_w[m].unsqueeze(0);
            e_wwt.index_put_({Slice(), Slice(), m}, sigw + torch::mm(dt.t(), dt));
        }
    }

    void _update_tau(torch::Tensor Y, int M, int N) {
        e = a + N * M * 1.0 / 2;
        double outer_expect = 0;
        for(auto& n : range(N, 0)) {
            for(auto& m : range(M, 0)) {
            	torch::Tensor dt = e_X.index({n, Slice()}).unsqueeze(0);
                outer_expect = outer_expect
                		+ torch::trace(e_wwt.index({Slice(), Slice(), m}).mm(sigx)).data().item<double>()
						+ (dt.mm(e_wwt.index({Slice(), Slice(), m})).mm(dt.t())[0][0]).data().item<double>();
            }
        }
        f = b + 0.5 * torch::sum(Y.pow(2)).data().item<double>() -
        		torch::sum(Y * e_w.mm(e_X.t()).t()).data().item<double>() + 0.5 * outer_expect;
        e_tau = e / f;
        e_log_tau = torch::mean(torch::log(rgamma(e, 1 / f, 1000))).data().item<double>();
    }

    double lower_bound(torch::Tensor Y, int M, int N, int D) {

        double LB = a * std::log(b) + (a - 1) * e_log_tau - b * e_tau - std::lgamma(a);
        LB = LB - (e * std::log(f) + (e - 1) * e_log_tau - f * e_tau - std::lgamma(e));

        for(auto& n : range(N, 0)) {
        	torch::Tensor dt = e_X.index({n, Slice()}).unsqueeze(0);
            LB = LB + (-(D * 0.5) * std::log(2 * M_PI) - 0.5 * (
                        torch::trace(sigx) + dt.mm(dt.t())[0][0]).data().item<double>());
            LB = LB - (-(D * 0.5) * std::log(2 * M_PI) -
            		  0.5 * torch::log(torch::linalg_det(sigx)).data().item<double>() - 0.5 * D);
        }

        for(auto& m : range(M, 0)) {
        	torch::Tensor dt = e_w.index({m, Slice()}).unsqueeze(0);
            LB = LB + (-(D * 0.5) * std::log(2 * M_PI) - 0.5 * (
                        torch::trace(sigw) + dt.mm(dt.t())[0][0]).data().item<double>());
            LB = LB - (-(D * 0.5) * std::log(2 * M_PI) -
            		  0.5 * torch::log(torch::linalg_det(sigw)).data().item<double>() - 0.5 * D);
        }

        double outer_expect = 0;
        for(auto& n : range(N, 0)) {
            for(auto& m : range(M, 0)) {
            	torch::Tensor dt = e_X.index({n, Slice()}).unsqueeze(0);
                outer_expect = outer_expect
                               + torch::trace(e_wwt.index({Slice(), Slice(), m}).mm(sigx)).data().item<double>()
                               + (dt.mm(e_wwt.index({Slice(), Slice(), m})).mm(dt.t())[0][0]).data().item<double>();
            }
        }

        LB = LB + (
                    -(N * M * 1.0 / 2) * std::log(2 * M_PI) + (N * M * 1.0 / 2) * e_log_tau
                    - 0.5 * e_tau * (torch::sum(Y.pow(2)) - 2 * torch::sum(Y * e_w.mm(e_X.t()).t()) + outer_expect).data().item<double>());
        return LB;
    }

    void _update(torch::Tensor Y, int N, int M, int D) {
        _update_X(Y, N, D);
        _update_W(Y, M, D);
        _update_tau(Y, M, N);
        double LB = lower_bound(Y, M, N, D);
        lbs.push_back(LB);
        if( verbose )
            printf("Lower bound: %.4f\n", LB);
    }

    void fit(torch::Tensor Y) {
        int N = Y.size(0), M = Y.size(1);
        //         temporarily comment these two lines out
        //         if not D:
        //             D = N
        _init_paras(N, M, D);

        for(auto& it : range(n_iters, 0)) {
            _update(Y, N, M, D);
            if(verbose)
                printf("Iters:%4d ", it);
            if(it >= 1) {
                if(std::abs(lbs[it] - lbs[it - 1]) < tol)
                    break;
            }
        }
    }

    torch::Tensor recover(void) {
        return e_X.mm(e_w.t());
    }

    std::vector<double> get_lbs(void) {
    	return lbs;
    }

    torch::Tensor get_e_X(void) {
    	return e_X;
    }

private:
	int D, n_iters;
	bool verbose;
	double a = 0., b = 0., e_tau = 0., tol = 0., e = 0., f = 0., e_log_tau = 0.;
	torch::Tensor e_w, e_wwt, e_X, e_XXt, sigx, sigw;
	std::vector<double> lbs;
};



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << std::lgamma(1.0) << '\n';
	std::cout << std::lgamma(281.0) << '\n';

	std::cout << at::lgamma(torch::tensor(281.0)) << '\n';
	torch::Tensor dt = rgamma(281.0, 1 / 281.13010831368626, 10);
	std::cout << torch::mean(torch::log(dt)) << '\n';

	torch::Tensor Y = torch::cat(
			{torch::randn({20, 2}),
			 torch::randn({20, 2}) + 5,
			 torch::randn({20, 2}) + torch::tensor({5., 0.}),
			 torch::randn({20, 2}) + torch::tensor({0., 5.})});
	std::cout << Y.sizes() << '\n';

	int N = Y.size(0);
	Y = torch::cat({Y, torch::randn({N, 5})}, 1);
	std::cout << Y.sizes() << '\n';

	std::cout << Y.index({Slice(), 0}) << '\n';
	torch::Tensor st = torch::stack({Y.index({Slice(), 0}), Y.index({Slice(), 0})}, 0);
	std::cout << st << '\n';
	std::optional<long int> dm = {0};
	torch::Tensor rt = torch::repeat_interleave(Y.index({Slice(), 0}).unsqueeze(0), 2, dm);
	std::cout << rt << '\n';

	//t = np.vstack([repmat(i, 20, 1) for i in range(1,5)])

	PPCA ppca(2, 100, true);
	ppca.fit(Y);

	int n = 0;
	std::vector<double> x, y;
	for(auto& i : ppca.get_lbs() ) {
		if(i != 0) {
			x.push_back(n * 1.0);
			y.push_back(i);
			n++;
		}
	}

	matplot::plot(x, y)->line_width(2.5);
	matplot::show();

	torch::Tensor e_X =  ppca.get_e_X();
	std::vector<double> c;
	for(int i = 1; i <= 5; i++) {
		for(int j = 0; j < 20; j++)
			c.push_back(i*1.0);
	}

	matplot::scatter(tensorTovector(e_X.index({Slice(), 0}).to(torch::kDouble)),
			tensorTovector(e_X.index({Slice(), 1}).to(torch::kDouble)), 15, c)->marker_face(true);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}

