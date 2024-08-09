/*
 * ComputerVision.cpp
 *
 *  Created on: Jul 20, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <torch/script.h>
#include <filesystem>

#include "../../Utils/opencv_helpfunctions.h"
#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor preprocess_image(cv::Mat mat, std::vector<int> nSize) {

	// Mean and std list for channels (Imagenet)
	std::vector<float> _mean = {0.485, 0.456, 0.406};
	std::vector<float> _std = {0.229, 0.224, 0.225};

	// Resize image
	if( nSize.size() > 0 )
		cv::resize(mat, mat, cv::Size(nSize[0], nSize[1]));

	torch::Tensor imgT = CvMatToTensor(mat, {}, false);

 	 // Normalize the channels
	imgT = NormalizeTensor(imgT, _mean, _std);

	// Add one more channel to the beginning. Tensor shape = 1,3,224,224
	imgT = imgT.unsqueeze(0);

	return imgT.requires_grad_(true);
}

cv::Mat recreate_image(torch::Tensor imgT) {
    //Recreates images from a torch variable, sort of reverse preprocessing

	std::vector<float> reverse_mean = {-0.485, -0.456, -0.406};
	std::vector<float> reverse_std = {1/0.229, 1/0.224, 1/0.225};

	imgT = deNormalizeTensor(imgT, reverse_mean, reverse_std);
	imgT.masked_fill_(imgT < 0., 0.);
	imgT.masked_fill_(imgT > 1.0, 1.);

	cv::Mat recreated_im = TensorToCvMat(imgT, true, false);
    //recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im;
}

class ClassSpecificImageGeneration {
public:
    // Produces an image that maximizes a certain class with gradient ascent

	ClassSpecificImageGeneration(torch::jit::script::Module _model, int _target_class) {
        _mean = {-0.485, -0.456, -0.406};
        _std = {1/0.229, 1/0.224, 1/0.225};
        model = _model;
        model.eval();
        target_class = _target_class;

        // Generate a random image
        torch::Tensor rand_img = torch::randint(0, 255, {3, 224, 224}).to(torch::kByte);
        created_image = TensorToCvMat(rand_img, false, false);

        // Create the folder to export images if not exists
        if( ! std::filesystem::exists("./src/Deep_learning/Ch12_Applications/generated") )
        	std::filesystem::create_directory("./src/Deep_learning/Ch12_Applications/generated");
	}

    std::tuple<std::vector<std::string>, std::vector<cv::Mat>> generate(int iterations=160) {
        //Generates class specific image
    	std::vector<cv::Mat> mats;
    	std::vector<std::string> tlts;

        float initial_learning_rate = 6;

        // Process image and return variable
        processed_image = preprocess_image(created_image, {});

        // Define optimizer for the image
        auto optimizer = torch::optim::SGD({processed_image}, initial_learning_rate);

        for(auto& i : range((iterations + 1), 1) ) {
        	// Zero grads
        	optimizer.zero_grad();

            // Forward
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(processed_image);
            torch::Tensor output = model.forward(inputs).toTensor();

            // Target specific class
            auto class_loss = -1*output.index({0, target_class});

            if( i % 30 == 0 || i == iterations ) {
                printf("Iteration: %d\tLoss: %.2f\n", i, class_loss.data().item<float>());
            }

            // Backward
            class_loss.backward();
            // Update image
            optimizer.step();
            // Recreate image
            created_image = recreate_image(processed_image.squeeze());

            if(i % 30 == 0 || i == iterations) {
                // Save image
                std::string im_path = "./src/Deep_learning/Ch12_Applications/generated/class_"+
				std::to_string(target_class)+"_"+"iter_"+std::to_string(i)+".png";
                cv::imwrite(im_path, created_image);
                mats.push_back(created_image.clone());
                tlts.push_back("c-"+std::to_string(target_class)+"-"+"iter-"+std::to_string(i));
            }
        }
        return std::make_tuple(tlts, mats);
    }

    torch::Tensor getProcessedImage() {
    	return processed_image;
    }

private:
	std::vector<float> _mean, _std;
	torch::jit::script::Module model;
	int target_class = 0;
	cv::Mat created_image;
	torch::Tensor processed_image;
 };



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

    int target_class = 130;  // Flamingo
    // Define model
	torch::jit::script::Module model;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		model = torch::jit::load("./data/AlexNet_jit_model.pt");
	} catch (const c10::Error& e) {
		std::cerr << "error loading the model: " << e.backtrace() << "\n";
		exit(-1);
	}
	model.to(torch::kCPU);
	/*
    for(torch::jit::Named<at::Tensor> p : model.named_parameters(true)) {
    	std::cout <<  p.value.defined() << " " << p.value.sizes() << p.value.grad() << '\n';
    	if( p.value.defined() )
    		p.value.detach().zero_();
    }
    */

	ClassSpecificImageGeneration csig = ClassSpecificImageGeneration(model, target_class);
	std::tuple<std::vector<std::string>, std::vector<cv::Mat>> rt =csig.generate();
	std::vector<std::string> tlts = std::get<0>(rt);
	std::vector<cv::Mat> mats = std::get<1>(rt);

	auto f = figure(true);
	f->width(f->width() * 3);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	for(int i = 0; i < 6; i++) {
		std::vector<std::vector<std::vector<unsigned char>>> img = CvMatToMatPlotVec(mats[i].clone(),
		    												{}, false, true);
		matplot::subplot(2, 3, i);
		matplot::title(tlts[i].c_str());
		matplot::imshow(img);
	}
	matplot::show();


	std::cout << "Done!\n";
	return 0;
}




