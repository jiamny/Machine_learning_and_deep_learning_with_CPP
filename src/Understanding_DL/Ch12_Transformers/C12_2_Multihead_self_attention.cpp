/*
 * C12_2_ Multihead_self_attention.cpp
 *
 *  Created on: Mar 12, 2025
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>

#include "../../Utils/activation.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

// Define softmax operation that works independently on each column
torch::Tensor softmax_cols(torch::Tensor data_in) {
    // Exponentiate all of the values
	torch::Tensor exp_values = torch::exp(data_in);

    // Sum over columns
	c10::OptionalArrayRef<long int> dim = {0};
	torch::Tensor denom = torch::sum(exp_values, dim);

    // Compute softmax (numpy broadcasts denominator to all rows automatically)
	torch::Tensor softmax = exp_values / denom;
    // return the answer
    return softmax;
}

// Now let's compute self attention in matrix form
torch::Tensor multihead_scaled_self_attention(torch::Tensor X, torch::Tensor omega_v1, torch::Tensor omega_q1,
		torch::Tensor omega_k1, torch::Tensor beta_v1, torch::Tensor beta_q1, torch::Tensor beta_k1,
		torch::Tensor omega_v2, torch::Tensor omega_q2, torch::Tensor omega_k2, torch::Tensor beta_v2,
		torch::Tensor beta_q2, torch::Tensor beta_k2, torch::Tensor omega_c) {

    // Write the multihead scaled self-attention mechanism.
	torch::Tensor Q1 = beta_q1 + torch::matmul(omega_q1, X);
	torch::Tensor  K1 = beta_k1 + torch::matmul(omega_k1, X);
	torch::Tensor  V1 = beta_v1 + torch::matmul(omega_v1, X);

	torch::Tensor A1 = torch::matmul(K1.t(), Q1);
	int Dq1 = Q1.size(0);
	A1 = A1 / std::ceil(std::sqrt(Dq1 * 1.0));

	torch::Tensor Aw1 = softmax_cols(A1);
	torch::Tensor x_prime1 = torch::matmul(V1, Aw1);

	torch::Tensor Q2 = beta_q2 + torch::matmul(omega_q2, X);
	torch::Tensor K2 = beta_k2 + torch::matmul(omega_k2, X);
	torch::Tensor V2 = beta_v2 + torch::matmul(omega_v2, X);

	torch::Tensor A2 = torch::matmul(K2.t(), Q2);
	int Dq2 = Q2.size(0);
	A2 = A2 / std::ceil(std::sqrt(Dq2 * 1.0));

	torch::Tensor Aw2 = softmax_cols(A2);
	torch::Tensor x_prime2 = torch::matmul(V2, Aw2);

	torch::Tensor Mx_prime = torch::cat({x_prime1, x_prime2}, 0);
	torch::Tensor X_prime = torch::matmul(omega_c, Mx_prime);
	return X_prime;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	// Number of inputs
	int N = 6;
	// Number of dimensions of each input
	int D = 8;

	// Choose random values for the parameters for the first head
	torch::Tensor X = torch::tensor(
			{{ 1.78862847,  0.43650985,  0.09649747, -1.8634927,  -0.2773882,  -0.35475898},
			 {-0.08274148, -0.62700068, -0.04381817, -0.47721803, -1.31386475,  0.88462238},
			 { 0.88131804,  1.70957306,  0.05003364, -0.40467741, -0.54535995, -1.54647732},
			 { 0.98236743, -1.10106763, -1.18504653, -0.2056499,   1.48614836,  0.23671627},
			 {-1.02378514, -0.7129932,   0.62524497, -0.16051336, -0.76883635, -0.23003072},
			 { 0.74505627,  1.97611078, -1.24412333, -0.62641691, -0.80376609, -2.41908317},
			 {-0.92379202, -1.02387576,  1.12397796, -0.13191423, -1.62328545,  0.64667545},
			 {-0.35627076, -1.74314104, -0.59664964, -0.58859438, -0.8738823,  0.02971382}});

	torch::Tensor omega_q1 = torch::tensor(
			{{ 1.76405235,  0.40015721,  0.97873798,  2.2408932,   1.86755799, -0.97727788, 0.95008842, -0.15135721},
			 {-0.10321885,  0.4105985,   0.14404357,  1.45427351,  0.76103773,  0.12167502, 0.44386323,  0.33367433},
			 { 1.49407907, -0.20515826,  0.3130677,  -0.85409574, -2.55298982,  0.6536186, 0.8644362,  -0.74216502},
			 { 2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877, 0.15494743,  0.37816252}});
	torch::Tensor omega_k1 = torch::tensor(
			{{-0.88778575, -1.98079647, -0.34791215,  0.15634897,  1.23029068,  1.20237985, -0.38732682, -0.30230275},
			 {-1.04855297, -1.42001794, -1.70627019,  1.9507754,  -0.50965218, -0.4380743, -1.25279536,  0.77749036},
			 {-1.61389785, -0.21274028, -0.89546656,  0.3869025,  -0.51080514, -1.18063218, -0.02818223,  0.42833187},
			 { 0.06651722,  0.3024719,  -0.63432209, -0.36274117, -0.67246045, -0.35955316, -0.81314628, -1.7262826 }});
	torch::Tensor omega_v1 = torch::tensor(
			{{ 0.17742614, -0.40178094, -1.63019835,  0.46278226, -0.90729836,  0.0519454, 0.72909056,  0.12898291},
			 { 1.13940068, -1.23482582,  0.40234164, -0.68481009, -0.87079715, -0.57884966, -0.31155253,  0.05616534},
			 { -1.16514984,  0.90082649,  0.46566244, -1.53624369,  1.48825219,  1.89588918, 1.17877957, -0.17992484},
			 {-1.07075262,  1.05445173, -0.40317695,  1.22244507,  0.20827498,  0.97663904, 0.3563664,   0.70657317}});
	torch::Tensor beta_q1 = torch::tensor(
			{{0.01050002},
			 {1.78587049},
			 {0.12691209},
			 {0.40198936}});
	torch::Tensor beta_k1 = torch::tensor(
			{{ 1.8831507 },
			 {-1.34775906},
			 {-1.270485  },
			 { 0.96939671}});
	torch::Tensor beta_v1 = torch::tensor(
			{{-1.17312341},
			 { 1.94362119},
			 {-0.41361898},
			 {-0.74745481}});

	// Choose random values for the parameters for the second head
	torch::Tensor omega_q2 = torch::tensor(
			{{ 1.92294203,  1.48051479,  1.86755896,  0.90604466, -0.86122569,  1.91006495,
			  -0.26800337,  0.8024564 },
			 { 0.94725197, -0.15501009,  0.61407937,  0.92220667,  0.37642553, -1.09940079,
			   0.29823817,  1.3263859 },
			 {-0.69456786, -0.14963454, -0.43515355,  1.84926373,  0.67229476,  0.40746184,
			  -0.76991607,  0.53924919},
			 {-0.67433266,  0.03183056, -0.63584608,  0.67643329,  0.57659082, -0.20829876,
			   0.39600671, -1.09306151}});
	torch::Tensor omega_k2 = torch::tensor(
			{{-1.49125759,  0.4393917,   0.1666735,   0.63503144,  2.38314477,  0.94447949,
			  -0.91282223,  1.11701629},
			 {-1.31590741, -0.4615846,  -0.06824161,  1.71334272, -0.74475482, -0.82643854,
			  -0.09845252, -0.66347829},
			 { 1.12663592, -1.07993151, -1.14746865, -0.43782004, -0.49803245,  1.92953205,
			   0.94942081,  0.08755124},
			 {-1.22543552,  0.84436298, -1.00021535, -1.5447711,   1.18802979,  0.31694261,
			   0.92085882,  0.31872765}});
	torch::Tensor omega_v2 = torch::tensor(
			{{ 0.85683061, -0.65102559, -1.03424284,  0.68159452, -0.80340966, -0.68954978,
			  -0.4555325,   0.01747916},
			 {-0.35399391, -1.37495129, -0.6436184,  -2.22340315,  0.62523145, -1.60205766,
			  -1.10438334,  0.05216508},
			 {-0.739563,    1.5430146, -1.29285691,  0.26705087, -0.03928282, -1.1680935,
			   0.52327666, -0.17154633},
			 { 0.77179055,  0.82350415,  2.16323595,  1.33652795, -0.36918184, -0.23937918,
			   1.0996596,   0.65526373}});
	torch::Tensor beta_q2 = torch::tensor(
			{{ 0.64013153},
			 {-1.61695604},
			 {-0.02432612},
			 {-0.73803091}});
	torch::Tensor beta_k2 = torch::tensor(
			{{ 0.2799246 },
			 {-0.09815039},
			 { 0.91017891},
			 { 0.31721822}});
	torch::Tensor beta_v2 = torch::tensor(
			{{ 0.78632796},
			 {-0.4664191 },
			 {-0.94444626},
			 {-0.41004969}});

	// Choose random values for the parameters
	torch::Tensor omega_c = torch::tensor(
			{{-0.01702041,  0.37915174,  2.25930895, -0.04225715, -0.955945,   -0.34598178,
			  -0.46359597,  0.48148147},
			 {-1.54079701,  0.06326199,  0.15650654,  0.23218104, -0.59731607, -0.23792173,
			  -1.42406091, -0.49331988},
			 {-0.54286148,  0.41605005, -1.15618243,  0.7811981,   1.49448454, -2.06998503,
			   0.42625873,  0.67690804},
			 {-0.63743703, -0.39727181, -0.13288058, -0.29779088, -0.30901297, -1.67600381,
			   1.15233156,  1.07961859},
			 {-0.81336426, -1.46642433,  0.52106488, -0.57578797,  0.14195316, -0.31932842,
			   0.69153875,  0.69474914},
			 {-0.72559738, -1.38336396, -1.5829384,   0.61037938, -1.18885926, -0.50681635,
			  -0.59631404, -0.0525673 },
			 {-1.93627981,  0.1887786,   0.52389102,  0.08842209, -0.31088617,  0.09740017,
			   0.39904635, -2.77259276},
			 { 1.95591231,  0.39009332, -0.65240858, -0.39095338,  0.49374178, -0.11610394,
			  -2.03068447,  2.06449286}});

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Compute multi-head scaled self attention\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	torch::Tensor X_prime = multihead_scaled_self_attention(X,omega_v1, omega_q1, omega_k1, beta_v1,
			beta_q1, beta_k1, omega_v2, omega_q2, omega_k2, beta_v2, beta_q2, beta_k2, omega_c);

	printf("Your answer:\n");
	print(X_prime);

	printf("\nTrue values:\n");
	printf("[[-21.207  -5.373 -20.933  -9.179 -11.319 -17.812]\n");
	printf(" [ -1.995   7.906 -10.516   3.452   9.863  -7.24 ]\n");
	printf(" [  5.479   1.115   9.244   0.453   5.656   7.089]\n");
	printf(" [ -7.413  -7.416   0.363  -5.573  -6.736  -0.848]\n");
	printf(" [-11.261  -9.937  -4.848  -8.915 -13.378  -5.761]\n");
	printf(" [  3.548  10.036  -2.244   1.604  12.113  -2.557]\n");
	printf(" [  4.888  -5.814   2.407   3.228  -4.232   3.71 ]\n");
	printf(" [  1.248  18.894  -6.409   3.224  19.717  -5.629]]\n");


	std::cout << "Done!\n";
	return 0;
}



