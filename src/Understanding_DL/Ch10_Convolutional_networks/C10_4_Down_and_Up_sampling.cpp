/*
 * C10_Downsampling_and_upsampling.cpp
 *
 *  Created on: Jan 31, 2025
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

torch::Tensor subsample(torch::Tensor x_in) {
	int r = int(std::ceil(x_in.size(0)/2.0));
	int c = int(std::ceil(x_in.size(1)/2.0));
	torch::Tensor x_out = torch::zeros({ r, c });
    // write the subsampling routine
	for(auto& i : range(r, 0)) {
		for(auto& j : range(c, 0)) {
			x_out.index_put_({i, j}, x_in.index({2*i, 2*j}));
		}
	}
    return x_out;
}

// Now let's try max-pooling
torch::Tensor maxpool(torch::Tensor x_in) {
	int r = int(std::floor(x_in.size(0)/2.0));
	int c = int(std::floor(x_in.size(1)/2.0));
	torch::Tensor x_out = torch::zeros({ r, c });
    // write the maxpool routine
	for(auto& i : range(r, 0)) {
		for(auto& j : range(c, 0)) {
			torch::Tensor t = x_in.index({Slice(2*i, 2*i+2), Slice(2*j, 2*j+2)});
			x_out.index_put_({i, j}, torch::max(t));
		}
	}
    return x_out;
}

torch::Tensor meanpool(torch::Tensor x_in) {
	int r = int(std::floor(x_in.size(0)/2.0));
	int c = int(std::floor(x_in.size(1)/2.0));
	torch::Tensor x_out = torch::zeros({ r, c });
    // write the maxpool routine
	for(auto& i : range(r, 0)) {
		for(auto& j : range(c, 0)) {
			torch::Tensor t = x_in.index({Slice(2*i, 2*i+2), Slice(2*j, 2*j+2)});
			x_out.index_put_({i, j}, torch::mean(t.to(torch::kFloat32)));
		}
	}
    return x_out;
}

// Let's first use the duplication method
torch::Tensor duplicate(torch::Tensor x_in) {
	int r = x_in.size(0);
	int c = x_in.size(1);
	torch::Tensor x_out = torch::zeros({r*2, c*2});
    // write the duplication routine
	for(auto& i : range(r, 0)) {
		for(auto& j : range(c, 0)) {
			x_out.index_put_({Slice(2*i, 2*i+2), Slice(2*j, 2*j+2)}, x_in[i][j]);
		}
	}
    return x_out;
}

torch::Tensor max_unpool(torch::Tensor x_in, torch::Tensor x_high_res) {
	int r = x_in.size(0);
	int c = x_in.size(1);
	torch::Tensor x_out = torch::zeros({r*2, c*2});
	// write the max_unpool routine
	for(auto& i : range(r, 0)) {
		for(auto& j : range(c, 0)) {
			torch::Tensor t = x_high_res.index({Slice(2*i, 2*i+2), Slice(2*j, 2*j+2)});
			// 获取最大值的索引
			torch::Tensor idx = (t==torch::max(t)).nonzero();
			int n = idx.size(0);
			for(auto& k :range(n, 0)) {
				x_out.index_put_({idx[k][0].data().item<long>() + 2*i, idx[k][1].data().item<long>() + 2*j}, x_in[i][j]);
			}
		}
	}
    return x_out;
}

torch::Tensor bilinear(torch::Tensor x_in) {
	int r = x_in.size(0);
	int c = x_in.size(1);
	torch::Tensor x_out = torch::zeros({ r*2, c*2 });
	torch::Tensor x_in_pad = torch::zeros({r+1, c+1});
	x_in_pad.index_put_({Slice(0, r), Slice(0, c)}, x_in);
	for(auto& i : range(2*r, 0)) {
		for(auto& j : range(2*c, 0)) {
			int m = 0;
			int n = 0;
			if( i % 2 == 0) {
				m = i / 2;
				if( j % 2 == 0) {
					n = j / 2;
					x_out.index_put_({i, j}, x_in_pad[m][n]);
				} else {
					int j_1 = static_cast<int>(j/2.0 - 0.5);
					int j_2 = static_cast<int>(j/2.0 + 0.5);
					x_out.index_put_({i, j}, torch::ceil((x_in_pad[m][j_1] + x_in_pad[m][j_2])/2.0));
				}
			} else {
				if( j % 2 == 0) {
					n = j / 2;
					int i_1 = static_cast<int>(i/2.0 - 0.5);
					int i_2 = static_cast<int>(i/2.0 + 0.5);
					x_out.index_put_({i, j}, torch::ceil((x_in_pad[i_1][n] + x_in_pad[i_2][n])/2.0));
				} else {
					int i_1 = static_cast<int>(i/2.0 - 0.5);
					int i_2 = static_cast<int>(i/2.0 + 0.5);
					int j_1 = static_cast<int>(j/2.0 - 0.5);
					int j_2 = static_cast<int>(j/2.0 + 0.5);
					x_out.index_put_({i, j},
							torch::ceil((x_in_pad[i_1][j_1] + x_in_pad[i_1][j_2] +x_in_pad[i_2][j_1] + x_in_pad[i_2][j_2])/4.0));
				}
			}
		}
	}

	return x_out;
}


void show_image(cv::Mat mat, std::string tlt, int w=500, int h=500) {
	cv::resize(mat, mat, cv::Size(w, h), cv::INTER_AREA);
    cv::imshow(tlt, mat);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Mat tensrTocvmat(torch::Tensor Subsampled_tensor) {
	int64_t height = Subsampled_tensor.size(0);
	int64_t width = Subsampled_tensor.size(1);
	auto tensor = Subsampled_tensor.reshape({width * height * 1});
	cv::Mat rev_mat(cv::Size(width, height), CV_8UC1, tensor.data_ptr());
	return rev_mat.clone();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	bool plt = false;

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Check Sub-sampling\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	// Define 4 by 4 original patch
	torch::Tensor orig_4_4 = torch::tensor({{1, 3, 5,3 }, {6,2,0,8}, {4,6,1,4}, {2,8,0,3}});
	printf("Original:\n");
	print(orig_4_4);

	printf("Subsampled:\n");
	print(subsample(orig_4_4));

	std::cout << "\n// ------------------------------------------------------------------------\n";
	std::cout << "// Load image\n";
	std::cout << "// ------------------------------------------------------------------------\n";
    cv::Mat mat;

    mat = cv::imread("data/test_image.png", cv::IMREAD_GRAYSCALE);
    std::cout << mat.channels() << '\n';
    if(plt) show_image(mat, "Gray Image");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Subsampled Image 1\n";
	std::cout << "// ------------------------------------------------------------------------\n";
    torch::Tensor img_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols}, c10::TensorOptions(torch::kByte));
    print(img_tensor);
    torch::Tensor Subsampled_tensor = subsample(img_tensor.squeeze());
    print(Subsampled_tensor);
    mat = tensrTocvmat(Subsampled_tensor);
    if(plt) show_image(mat, "Subsampled Image 1");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Subsampled Image 2\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor Subsampled_tensor2 = subsample(Subsampled_tensor.squeeze());
    mat = tensrTocvmat(Subsampled_tensor2);
    if(plt) show_image(mat, "Subsampled Image 2");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Subsampled Image 3\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor Subsampled_tensor3 = subsample(Subsampled_tensor2.squeeze());
    mat = tensrTocvmat(Subsampled_tensor3);
    if(plt) show_image(mat, "Subsampled Image 3");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Check Max-pooling\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	printf("Original:\n");
	print(orig_4_4);

	printf("Max-pooled:\n");
	print(maxpool(orig_4_4));

	std::cout << "\n// ------------------------------------------------------------------------\n";
	std::cout << "// Max-pooled Image\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	mat = cv::imread("data/test_image.png", cv::IMREAD_GRAYSCALE);
	if(plt) show_image(mat, "Original Image");

	img_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols}, c10::TensorOptions(torch::kByte)).clone();
	torch::Tensor maxpool_tensor = maxpool(img_tensor.squeeze());
	mat = tensrTocvmat(maxpool_tensor);
	if(plt) show_image(mat, "Maxpool Image 1");

	torch::Tensor maxpool_tensor2 = maxpool(maxpool_tensor.squeeze());
	mat = tensrTocvmat(maxpool_tensor2);
	if(plt) show_image(mat, "Maxpool Image 2");

	torch::Tensor maxpool_tensor3 = maxpool(maxpool_tensor2.squeeze());
	mat = tensrTocvmat(maxpool_tensor3);
	if(plt) show_image(mat, "Maxpool Image 3");

	std::cout << "\n// ------------------------------------------------------------------------\n";
	std::cout << "// Mean-pooled Image\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	mat = cv::imread("data/test_image.png", cv::IMREAD_GRAYSCALE);
	if(plt) show_image(mat, "Original Image");

	img_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols}, c10::TensorOptions(torch::kByte)).clone();
	torch::Tensor mean_tensor = meanpool(img_tensor.squeeze());
	mat = tensrTocvmat(mean_tensor);
	if(plt) show_image(mat, "Meanpool Image 1");

	torch::Tensor mean_tensor2 = meanpool(mean_tensor.squeeze());
	mat = tensrTocvmat(mean_tensor2);
	if(plt) show_image(mat, "Meanpool Image 2");

	torch::Tensor mean_tensor3 = meanpool(mean_tensor2.squeeze());
	mat = tensrTocvmat(mean_tensor3);
	if(plt) show_image(mat, "Meanpool Image 3");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Up-sampling\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	torch::Tensor orig_2_2 = torch::tensor({{6, 8}, {8,4}});

	printf("Original:\n");
	print(orig_2_2);
	printf("\nDuplicated:\n");
	print(duplicate(orig_2_2));
	printf("\n");

	// Let's re-upsample, sub-sampled rick
	torch::Tensor data_duplicate = duplicate(Subsampled_tensor3);

	mat = tensrTocvmat(data_duplicate);
	if(plt) show_image(mat, "data_duplicate");

	torch::Tensor data_duplicate2 = duplicate(data_duplicate);
	mat = tensrTocvmat(data_duplicate2);
	if(plt) show_image(mat, "data_duplicate2");

	torch::Tensor data_duplicate3 = duplicate(data_duplicate2);
	mat = tensrTocvmat(data_duplicate3);
	if(plt) show_image(mat, "data_duplicate3");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Max pooling\n";
	std::cout << "// ------------------------------------------------------------------------\n";

	printf("Original:\n");
	print(orig_2_2);
	printf("\nMax pooled:\n");
	print(max_unpool(orig_2_2, orig_4_4));
	printf("\n");

	torch::Tensor data_max_unpool = max_unpool(maxpool_tensor3, maxpool_tensor2);
	mat = tensrTocvmat(maxpool_tensor3);
	if(plt) show_image(mat, "Maxpool_tensor3");

	mat = tensrTocvmat(data_max_unpool);
	if(plt) show_image(mat, "Data_max_unpool");

	torch::Tensor  data_max_unpool2 = max_unpool(data_max_unpool, maxpool_tensor);
	mat = tensrTocvmat(data_max_unpool2);
	if(plt) show_image(mat, "Data_max_unpool2");

	torch::Tensor data_max_unpool3 = max_unpool(data_max_unpool2, img_tensor.squeeze());
	mat = tensrTocvmat(data_max_unpool3);
	if(plt) show_image(mat, "Data_max_unpool3");

	std::cout << "// ------------------------------------------------------------------------\n";
	std::cout << "// Bilinear up-sampling\n";
	std::cout << "// ------------------------------------------------------------------------\n";
	orig_2_2 = torch::tensor({{2, 4}, {4,8}});
	print(bilinear(orig_2_2));
	std::cout << '\n';

	std::cout << "Done!\n";
	return 0;
}



