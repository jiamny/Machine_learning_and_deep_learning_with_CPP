/*
 * opencv_helpfunctions.cpp
 *
 *  Created on: Jun 29, 2024
 *      Author: jiamny
 */

#include "opencv_helpfunctions.h"

torch::Tensor load_image(std::string path, std::vector<int> nSize) {
    cv::Mat mat;

    mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    int im_h = mat.rows, im_w = mat.cols, chs = mat.channels();

    cv::Mat Y;

    if( ! nSize.empty() ) {
        int h = nSize[0], w = nSize[1];

        float res_aspect_ratio = w*1.0/h;
        float input_aspect_ratio = im_w*1.0/im_h;

        int dif = im_w;
        if( im_h > im_w ) int dif = im_h;

        int interpolation = cv::INTER_CUBIC;
        if( dif > static_cast<int>((h+w)*1.0/2) ) interpolation = cv::INTER_AREA;

        if( input_aspect_ratio != res_aspect_ratio ) {
            if( input_aspect_ratio > res_aspect_ratio ) {
                int im_w_r = static_cast<int>(input_aspect_ratio*h);
                int im_h_r = h;

                cv::resize(mat, mat, cv::Size(im_w_r, im_h_r), (0,0), (0,0), interpolation);
                int x1 = static_cast<int>((im_w_r - w)/2);
                int x2 = x1 + w;
                mat(cv::Rect(x1, 0, w, im_h_r)).copyTo(Y);
            }

            if( input_aspect_ratio < res_aspect_ratio ) {
                int im_w_r = w;
                int im_h_r = static_cast<int>(w/input_aspect_ratio);
                cv::resize(mat, mat, cv::Size(im_w_r , im_h_r), (0,0), (0,0), interpolation);
                int y1 = static_cast<int>((im_h_r - h)/2);
                int y2 = y1 + h;
                mat(cv::Rect(0, y1, im_w_r, h)).copyTo(Y); // startX,startY,cols,rows
            }
        } else {
        	 cv::resize(mat, Y, cv::Size(w, h), interpolation);
        }
    } else {
    	Y = mat.clone();
    }

    torch::Tensor img_tensor = torch::from_blob(Y.data, {Y.channels(), Y.rows, Y.cols}, c10::TensorOptions(torch::kByte));
    return img_tensor;
}



