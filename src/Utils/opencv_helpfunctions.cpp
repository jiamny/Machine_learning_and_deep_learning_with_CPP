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

cv::Mat TensorToCvMat(torch::Tensor img, bool is_float, bool  toBGR) {

	if( is_float ) {
		float maxV = torch::max(img).data().item<float>();
		if( maxV > 1.0 ) img.div_(maxV);
	}

	torch::Tensor rev_tensor;
	if( is_float ) {
		auto data_out = img.contiguous().detach().clone();
		rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});
	} else {
		auto data_out = img.contiguous().detach().clone();
		rev_tensor = data_out.to(torch::kByte).permute({1, 2, 0});
	}

	// shape of tensor
	int64_t height = rev_tensor.size(0);
	int64_t width = rev_tensor.size(1);

	// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
	// so we must reshape tensor, otherwise we get a 3x3 grid
	auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

	// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
	cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
	cv::Mat rev_bgr_mat = rev_rgb_mat.clone();

	if( toBGR )
		cv::cvtColor(rev_bgr_mat, rev_bgr_mat, cv::COLOR_RGB2BGR);

	return rev_bgr_mat;
}


torch::Tensor CvMatToTensor(cv::Mat img, std::vector<int> img_size, bool toRGB) {
	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	if( toRGB )
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	if( img_size.size() > 0 ) {
		// ---------------------------------------------------------------
		// opencv resize the Size() - should be (width/cols x height/rows)
		// ---------------------------------------------------------------
		cv::resize(img, img, cv::Size(img_size[0], img_size[1]));
	}

	torch::Tensor imgT = torch::from_blob(img.data,
						{img.rows, img.cols, img.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

std::vector<std::vector<std::vector<unsigned char>>>  CvMatToMatPlotVec(cv::Mat img,
												std::vector<int> img_size, bool toRGB, bool is_float) {
	if( toRGB )
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	if( img_size.size() > 0 )
		cv::resize(img, img, cv::Size(img_size[0], img_size[1]));

	if( is_float ) {
		img *= 255;
		switch( img.channels() ) {
		case 1:
			img.convertTo(img, CV_8UC1);
			break;
		case 2:
			img.convertTo(img, CV_8UC2);
			break;
		case 3:
			img.convertTo(img, CV_8UC3);
			break;
		case 4:
			img.convertTo(img, CV_8UC4);
			break;
		}
	}

    std::vector<cv::Mat> channels(img.channels());
    cv::split(img, channels);

    std::vector<std::vector<std::vector<unsigned char>>> image;

    for(size_t i = 0; i < img.channels(); i++) {
    	std::vector<std::vector<unsigned char>> ch;
    	for(size_t j = 0; j < channels[i].rows; j++) {
    		std::vector<unsigned char>  r = channels[i].row(j).reshape(1, 1);
    		ch.push_back(r);
    	}
    	image.push_back(ch);
    }
    return image;
}


std::vector<std::vector<std::vector<unsigned char>>> tensorToMatrix4MatplotPP(torch::Tensor img,
																				bool is_float, bool need_permute ) {
	// OpenCV is BGR, Pillow is RGB
	torch::Tensor data_out = img.contiguous().detach().clone();
	torch::Tensor rev_tensor;
	if( is_float ) {
		if( need_permute )
			rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});
		else
			rev_tensor = data_out.mul(255).to(torch::kByte);
	} else {
		if( need_permute )
			rev_tensor = data_out.to(torch::kByte).permute({1, 2, 0});
		else
			rev_tensor = data_out.to(torch::kByte);
	}

	//std::vector<uint8_t> z(mimg.data_ptr<uint8_t>(), mimg.data_ptr<uint8_t>() + sizeof(uint8_t)*mimg.numel());
	//uint8_t* zptr = &(z[0]);
	// shape of tensor
	int64_t height = rev_tensor.size(0);
	int64_t width = rev_tensor.size(1);


	// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
	// so we must reshape tensor, otherwise we get a 3x3 grid
	auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

	cv::Mat Y;

	switch(static_cast<int>(rev_tensor.size(2))) {
		case 1: {
			cv::Mat rev_rgb_mat1(cv::Size(width, height), CV_8UC1, tensor.data_ptr());
			Y = rev_rgb_mat1.clone();
			break;
		}
		case 2: {
			cv::Mat rev_rgb_mat2(cv::Size(width, height), CV_8UC2, tensor.data_ptr());
			Y = rev_rgb_mat2.clone();
			break;
		}
		case 3: {
		// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
			cv::Mat rev_rgb_mat3(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
			Y = rev_rgb_mat3.clone();
			break;
		}
		case 4: {
			cv::Mat rev_rgb_mat4(cv::Size(width, height), CV_8UC4, tensor.data_ptr());
			Y = rev_rgb_mat4.clone();
			break;
		}
	}

	std::vector<std::vector<std::vector<unsigned char>>> image;

	std::vector<cv::Mat> channels(Y.channels());
	cv::split(Y, channels);

    for(size_t i = 0; i < Y.channels(); i++) {
    	std::vector<std::vector<unsigned char>> ch;
    	//std::cout   << channels[i].rows << '\n';
    	//std::cout   << channels[i].cols << '\n';
    	//std::cout   << channels[i].row(0).size() << '\n';
    	for(size_t j = 0; j < channels[i].rows; j++) {
    		std::vector<unsigned char>  r = channels[i].row(j).reshape(1, 1);
    		//std::cout   << r.size() << '\n';
    		ch.push_back(r);
    	}
    	image.push_back(ch);
    }

	return image;
}

torch::Tensor deNormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_) {

	torch::Tensor mean = torch::from_blob((float *)mean_.data(),
				{(long int)mean_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();  	// mean{C,1,1}
	torch::Tensor std = torch::from_blob((float *)std_.data(),
				{(long int)std_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();		// std{C,1,1}

	long int channels = imgT.size(0);

	torch::Tensor meanF = mean;
	if(channels < meanF.size(0)){
		meanF = meanF.split(/*split_size=*/channels, /*dim=*/0).at(0);  // meanF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor stdF = std;
	if(channels < stdF.size(0)){
		stdF = stdF.split(/*split_size=*/channels, /*dim=*/0).at(0);	// stdF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor data_out_src = imgT * stdF.to(imgT.device()) + meanF.to(imgT.device());  // data_in{C,H,W}, meanF{*,1,1}, stdF{*,1,1} ===> data_out_src{C,H,W}

	return data_out_src.contiguous().detach().clone();
}


torch::Tensor NormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_) {

	torch::Tensor mean = torch::from_blob((float *)mean_.data(),
				{(long int)mean_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();  	// mean{C,1,1}
	torch::Tensor std = torch::from_blob((float *)std_.data(),
				{(long int)std_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();		// std{C,1,1}

	long int channels = imgT.size(0);

	torch::Tensor meanF = mean;
	if(channels < meanF.size(0)){
		meanF = meanF.split(/*split_size=*/channels, /*dim=*/0).at(0);  // meanF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor stdF = std;
	if(channels < stdF.size(0)){
		stdF = stdF.split(/*split_size=*/channels, /*dim=*/0).at(0);	// stdF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor data_out_src = (imgT - meanF.to(imgT.device())) / stdF.to(imgT.device());  // data_in{C,H,W}, meanF{*,1,1}, stdF{*,1,1} ===> data_out_src{C,H,W}

	return data_out_src.contiguous().detach().clone();
}

