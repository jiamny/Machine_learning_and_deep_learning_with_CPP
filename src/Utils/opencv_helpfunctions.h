/*
 * opencv_helpfunctions.h
 *
 *  Created on: Jun 29, 2024
 *      Author: jiamny
 */

#ifndef OPENCV_HELPFUNCTIONS_H_
#define OPENCV_HELPFUNCTIONS_H_

#pragma once

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/utils.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <numeric>   // iota()
#include <random>
#include <tuple>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

torch::Tensor  CvMatToTensor(cv::Mat img, std::vector<int> img_size, bool toRGB = false);

cv::Mat TensorToCvMat( torch::Tensor img, bool is_float = true, bool toBGR = true );


torch::Tensor load_image(std::string path, std::vector<int> nSize = {});

torch::Tensor deNormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_);

torch::Tensor NormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_);

std::vector<std::vector<std::vector<unsigned char>>> tensorToMatrix4MatplotPP(torch::Tensor data,
																				bool is_float=true, bool need_permute=true);

std::vector<std::vector<std::vector<unsigned char>>>  CvMatToMatPlotVec(cv::Mat img,
														std::vector<int> img_size, bool toRGB = true, bool is_float = false);


#endif /* OPENCV_HELPFUNCTIONS_H_ */
