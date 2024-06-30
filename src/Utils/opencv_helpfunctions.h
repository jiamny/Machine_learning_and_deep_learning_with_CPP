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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

torch::Tensor load_image(std::string path, std::vector<int> nSize = {});

#endif /* OPENCV_HELPFUNCTIONS_H_ */
