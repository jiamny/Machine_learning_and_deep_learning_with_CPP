/*
 * C15_1_GAN_toy_example.cpp
 *
 *  Created on: Mar 25, 2025
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


// Get a batch of real data.  Our goal is to make data that looks like this.
torch::Tensor get_real_data_batch(int n_sample) {
	torch::Tensor x_true = torch::randn({1,n_sample}) + 7.5;
    return x_true;
}

// This is our generator -- takes the single parameter theta of the generative model and generates n samples
torch::Tensor generator(torch::Tensor z, float theta) {
	torch::Tensor x_gen = z + theta;
    return x_gen;
}


// Logistic sigmoid, maps from [-infty,infty] to [0,1]
torch::Tensor sig(torch::Tensor data_in) {
    return  1.0 / (1.0 + torch::exp(-data_in));
}

// Discriminator computes y
torch::Tensor discriminator(torch::Tensor x, float phi0, float phi1) {
    return sig(phi0 + phi1 * x);
}

// Draws a figure like Figure 15.1a
void draw_data_model(torch::Tensor x_real, torch::Tensor x_syn, float phi0=0., float phi1=0.) {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto fx = F->nexttile();
	matplot::hold(fx, true);

	int n = x_syn.size(1);
	for(auto& i : range( n, 0 )) {
		double x = x_syn[0][i].data().item<float>();
	    matplot::arrow(fx, x, 0., x, 0.33)->line_width(2).color("b");
	}

	n = x_real.size(1);
	for(auto& i : range( n, 0 )) {
		double x = x_real[0][i].data().item<float>();
	    matplot::arrow(fx, x, 0., x, 0.33)->line_width(2).color("m");
	}


	if( phi0 != 0.0) {
		torch::Tensor x_model = torch::arange(0., 10., 0.01);
		torch::Tensor y_model = discriminator(x_model, phi0, phi1);
		matplot::plot(fx, tensorTovector(x_model.to(torch::kDouble)),
				tensorTovector(y_model.to(torch::kDouble)), "k-:")->line_width(2);
	}
	matplot::xlim(fx, {0,10});
	matplot::ylim(fx, {0,1});

	matplot::show();

}

// Discriminator loss
torch::Tensor compute_discriminator_loss(torch::Tensor x_real, torch::Tensor x_syn, float phi0, float phi1) {

    /*  compute the loss for the discriminator
    # Run the real data and the synthetic data through the discriminator
    # Then use the standard binary cross entropy loss with the y=1 for the real samples
    # and y=0 for the synthesized ones.
    */
	torch::Tensor y_real = discriminator(x_real, phi0, phi1);
	torch::Tensor y_syn = discriminator(x_syn, phi0, phi1);
	torch::Tensor loss = -1.0*torch::sum(torch::log(1 - y_syn)) - torch::sum(torch::log(y_real));

    return loss;
}

// Gradient of loss (cheating, using finite differences)
std::tuple<float, float> compute_discriminator_gradient(torch::Tensor x_real, torch::Tensor x_syn, float phi0, float phi1) {
  float delta = 0.0001;
  torch::Tensor loss1 = compute_discriminator_loss(x_real, x_syn, phi0, phi1);
  torch::Tensor loss2 = compute_discriminator_loss(x_real, x_syn, phi0+delta, phi1);
  torch::Tensor loss3 = compute_discriminator_loss(x_real, x_syn, phi0, phi1+delta);
  float dl_dphi0 = (loss2.data().item<float>() - loss1.data().item<float>()) / delta;
  float dl_dphi1 = (loss3.data().item<float>() - loss1.data().item<float>()) / delta;

  return std::make_tuple(dl_dphi0, dl_dphi1);
}

// This routine performs gradient descent with the discriminator
std::tuple<float, float> update_discriminator(torch::Tensor x_real, torch::Tensor x_syn, int n_iter, float phi0, float phi1) {

  // Define learning rate
  float alpha = 0.01;
  float dl_dphi0 = 0., dl_dphi1 = 0.;

  // Get derivatives
  printf("Initial discriminator loss = %.4f\n", compute_discriminator_loss(x_real, x_syn, phi0, phi1).data().item<float>());
  for(auto& iter : range(n_iter, 0)) {
    // Get gradient
    std::tie(dl_dphi0, dl_dphi1) = compute_discriminator_gradient(x_real, x_syn, phi0, phi1);
    // Take a gradient step downhill
    phi0 = phi0 - alpha * dl_dphi0 ;
    phi1 = phi1 - alpha * dl_dphi1 ;
  }
  printf("Final Discriminator Loss = %.4f\n", compute_discriminator_loss(x_real, x_syn, phi0, phi1).data().item<float>());

  return std::make_tuple(phi0, phi1);
}

torch::Tensor compute_generator_loss(torch::Tensor z, float theta, float phi0, float phi1) {
	/*
	# Run the generator on the latent variables z with the parameters theta
	# to generate new data x_syn
	# Then run the discriminator on the new data to get the probability of being real
	# The loss is the total negative log probability of being synthesized (i.e. of not being real)
	*/
	torch::Tensor x_syn = generator(z, theta);
	torch::Tensor y_syn = discriminator(x_syn, phi0, phi1);
	torch::Tensor loss = -1.0 * torch::sum(torch::log(1 - y_syn));

	return loss;
}


torch::Tensor compute_generator_gradient(torch::Tensor z, float theta, float phi0, float phi1) {
  float delta = 0.0001;
  torch::Tensor loss1 = compute_generator_loss(z, theta, phi0, phi1);
  torch::Tensor loss2 = compute_generator_loss(z, theta+delta, phi0, phi1);
  torch::Tensor dl_dtheta = (loss2-loss1)/ delta;
  return dl_dtheta;
}

float update_generator(torch::Tensor z, float theta, int n_iter, float phi0, float phi1) {
    // Define learning rate
    float alpha = 0.02;

    // Get derivatives
    printf("Initial generator loss = %.4f\n", compute_generator_loss(z, theta, phi0, phi1).data().item<float>());
    for(auto& iter : range(n_iter, 0)) {
      // Get gradient
    	torch::Tensor dl_dtheta = compute_generator_gradient(z, theta, phi0, phi1);
      // Take a gradient step (uphill, since we are trying to make synthesized data less well classified by discriminator)
      theta = theta + alpha * dl_dtheta.data().item<float>();
    }
    printf("Final generator loss = %.4f\n", compute_generator_loss(z, theta, phi0, phi1).data().item<float>());
    return theta;
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available." : "Using CPU.") << '\n';

	// Get data batch
	torch::Tensor x_real = get_real_data_batch(10);

	// Initialize generator and synthesize a batch of examples
	float theta = 3.0;

	torch::Tensor z = torch::randn({1, 10});
	torch::Tensor x_syn = generator(z, theta);

	// Initialize discriminator model
	float phi0 = -2;
	float phi1 = 1;

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Draws a figure like Figure 15.1a\n";
	std::cout << "// --------------------------------------------------\n";
	draw_data_model(x_real, x_syn, phi0, phi1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Test the loss\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor loss = compute_discriminator_loss(x_real, x_syn, phi0, phi1);
	printf("True Loss = 13.641471, Your loss=%f\n", loss.data().item<float>() );

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Let's update the discriminator (sigmoid curve)\n";
	std::cout << "// --------------------------------------------------\n";
	int n_iter = 100;
	printf("Initial parameters (%f, %f)\n", phi0, phi1);
	std::tie(phi0, phi1) = update_discriminator(x_real, x_syn, n_iter, phi0, phi1);
	printf("Final parameters (%f, %f)\n", phi0, phi1);
	draw_data_model(x_real, x_syn, phi0, phi1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Test generator loss to check you have it correct\n";
	std::cout << "// --------------------------------------------------\n";
	loss = compute_generator_loss(z, theta, -2, 1);
	printf("True Loss = 13.578567, Your loss=%f\n", loss.data().item<float>() );


	n_iter = 10;
	theta = 3.0;
	printf("Theta before %f\n", theta);
	theta = update_generator(z, theta, n_iter, phi0, phi1);
	printf("Theta after %f\n", theta);

	x_syn = generator(z,theta);
	draw_data_model(x_real, x_syn, phi0, phi1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Now let's define a full GAN loop\n";
	std::cout << "// --------------------------------------------------\n";

	// Initialize the parameters
	theta = 3;
	phi0 = -2;
	phi1 = 1;

	// Number of iterations for updating generator and discriminator
	int n_iter_discrim = 300;
	int n_iter_gen = 3;

	printf("Final parameters (%f, %f)\n", phi0, phi1);
	for(auto& c_gan_iter : range(5, 0)) {

	  // Run generator to product synthesized data
	  x_syn = generator(z, theta);
	  draw_data_model(x_real, x_syn, phi0, phi1);

	  // Update the discriminator
	  printf("Updating discriminator\n");
	  std::tie(phi0, phi1) = update_discriminator(x_real, x_syn, n_iter_discrim, phi0, phi1);
	  draw_data_model(x_real, x_syn, phi0, phi1);

	  // Update the generator
	  printf("Updating generator\n");
	  theta = update_generator(z, theta, n_iter_gen, phi0, phi1);
	}

	std::cout << "Done!\n";
	return 0;
}





