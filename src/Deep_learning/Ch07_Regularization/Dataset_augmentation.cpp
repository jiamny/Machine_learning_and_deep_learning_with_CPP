/*
 * Dataset_augmentation.cpp
 *
 *  Created on: May 24, 2024
 *      Author: jiamny
 */
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Utils/fashion.h"

using torch::indexing::Slice;
using torch::indexing::None;

class Image {
public:
	Image(torch::Tensor _image) {
		image = _image.clone();
        row = image.size(0); // 图像高度
        col = image.size(1); // 图像宽度
        transform = torch::empty(0);
	}

	torch::Tensor Translation(int delta_x, int delta_y) {
       /*
          平移。
          参数说明：
        delta_x：控制左右平移，若大于0左移，小于0右移
        delta_y：控制上下平移，若大于0上移，小于0下移
       */
        transform = torch::tensor({{1, 0, delta_x},
                                   {0, 1, delta_y},
                                   {0,  0,  1}});
        return operate();
    }

    torch::Tensor Resize(int alpha){
        /*
          缩放。
          参数说明：
        alpha：缩放因子，不进行缩放设置为1
        */
        transform = torch::tensor({{alpha, 0, 0},
                                   {0, alpha, 0},
                                   {0,  0,  1}});
        return operate();
    }

    torch::Tensor HorMirror(void) {
        //水平镜像。
        transform = torch::tensor({{1,  0,  0},
                                   {0, -1, col-1},
                                   {0,  0,  1}});
        return operate();
    }

    torch::Tensor  VerMirror(void) {
    	// 垂直镜像。
        transform = torch::tensor({{-1, 0, row-1},
                                   {0,  1,  0},
                                   {0,  0,  1}});
        return operate();
    }

    torch::Tensor Rotate(double angle) {
    	/*
    		旋转。
       	参数说明：
        angle：旋转角度
        */
        transform = torch::tensor({{std::cos(angle), - std::sin(angle), 0},
                                   {std::sin(angle), std::cos(angle), 0},
								   {    0,              0,         1}});
        return operate();
    }

    torch::Tensor operate(void) {
    	torch::Tensor temp = torch::zeros(image.sizes(), image.dtype());
        for( auto& i : range(row, 0) ) {
            for( auto& j : range(col, 0) ) {
            	torch::Tensor temp_pos = torch::tensor({i, j, 1}).reshape({3, -1}).to(image.dtype());
                //[x,y,z] = torch::dot(transform, temp_pos);
                torch::Tensor xyz = torch::mm(transform.to(image.dtype()), temp_pos);
                int x = xyz[0].data().item<int>();
                int y = xyz[1].data().item<int>();

                if( x >= row || y >= col || x < 0 || y < 0 ) {
                	if( image.sizes().size() > 2 )
                		temp.index_put_({i, j, Slice()}, 0);
                	else
                		temp.index_put_({i, j}, 0);
                } else {
                    //temp[i,j,:] = image[x,y]
                	if( image.sizes().size() > 2 )
                		temp.index_put_({i, j, Slice()}, image.index({x, y, Slice()}));
                	else
                		temp.index_put_({i, j}, image.index({x, y}));
                }
            }
        }
        return temp;
    }

private:
	torch::Tensor image = torch::empty(0);
	int row = 0, col = 0;
	torch::Tensor transform = torch::empty(0);
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	const std::string FASHION_data_path("/media/hhj/localssd/DL_data/fashion_MNIST/");

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain);

	//auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
	//		                .map(torch::data::transforms::Stack<>());
	torch::Tensor images = train_dataset.images();
	std::cout << images.sizes() << " " << images.dtype() << '\n';

	torch::Tensor img = images[0].squeeze();
	img = img.mul(255).to(torch::kByte);

	Image Img(img.clone());
	torch::Tensor vimg = Img.VerMirror();

	cv::Mat im(cv::Size{ static_cast<int>(img.size(0)), static_cast<int>(img.size(1))}, CV_8UC1, img.data_ptr<uchar>());
	cv::resize(im, im, cv::Size(300, 300), cv::INTER_AREA);
	cv::imshow("orignal image", im.clone());
	if(cv::waitKey(0) == 27)
		cv::destroyAllWindows();

	cv::Mat verImg(cv::Size{ static_cast<int>(vimg.size(0)), static_cast<int>(vimg.size(1))}, CV_8UC1, vimg.data_ptr<uchar>());
	cv::resize(verImg, verImg, cv::Size(300, 300), cv::INTER_AREA);
	cv::imshow("VerMirror image", verImg.clone());
	if(cv::waitKey(0) == 27)
		cv::destroyAllWindows();

	std::cout << "Done!\n";
	return 0;
}
