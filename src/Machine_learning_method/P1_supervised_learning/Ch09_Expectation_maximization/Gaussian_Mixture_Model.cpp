/*
 * Gaussian_Mixture_Model.cpp
 *
 *  Created on: May 20, 2024
 *      Author: jiamny
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include "../../../Utils/csvloader.h"
#include "../../../Utils/TempHelpFunctions.h"
#include "../../../Utils/helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

class GMM {
public:
	GMM(int _k, int _max_epochs=100, double _tolerance=1e-8) {
        /*
        :param k: the number of clusters the algorithm will form.
        :param max_epochs: The number of iterations the algorithm will run for if it does
        not converge before that.
        :param tolerance: float
        If the difference of the results from one iteration to the next is
        smaller than this value we will say that the algorithm has converged.
        */
        k = _k;
        max_epochs = _max_epochs;
        tolerance = _tolerance;
        parameters.clear();
        responsibility = torch::empty(0);
        responsibilities.clear();
        sample_assignments = torch::empty(0);

	}

	torch::Tensor normalization(torch::Tensor X) {
	    /*
	    :param X: Input tensor
	    :return: Normalized input using l2 norm.
	    */
		const std::optional<c10::Scalar> p = {2};
		c10::ArrayRef<long int> dim = {-1};
		torch::Tensor l2 = torch::norm(X, p, dim);
	    l2.masked_fill_(l2 == 0, 1);
	    return X / l2.unsqueeze(1);
	}

	torch::Tensor covariance_matrix(torch::Tensor X) {
        /*
        :param X: Input tensor
        :return: cavariance of input tensor
        */
    	int64_t n_samples = X.size(0);

    	c10::OptionalArrayRef<long int> dm = {0};
    	torch::Tensor centering_X = X - X.mean(dm);
    	torch::Tensor covariance_matrix = torch::mm(centering_X.t(), centering_X)/(n_samples - 1);

        return covariance_matrix;
	}

	void random_gaussian_initialization(torch::Tensor X) {
        /*
        Since we are using iris dataset, we know the no. of class is 3.
        We create three gaussian distribution representing each class with
        random sampling of data to find parameters like μ and 𝚺/N (covariance matrix)
        for each class
        :param X: input tensor
        :return: 3 randomly selected mean and covariance of X, each act as a separate cluster
        */
        int n_samples = X.size(0);
        prior = (1.0 / k) * torch::ones(k);

        for(auto& cls : range(k, 0)) {
        	std::map<std::string, torch::Tensor> parameter;
            parameter["mean"] = X[torch::randperm(n_samples)[0].data().item<long>()];
            parameter["cov"] = covariance_matrix(X);
            parameters.push_back(parameter);
        }
	}

	torch::Tensor multivariate_gaussian_distribution(torch::Tensor X, std::map<std::string, torch::Tensor> parameters) {
        /*
        Checkout the equation from Multi-Dimensional Model from blog link posted above.
        We find the likelihood of each sample w.r.t to the parameters initialized above for each separate cluster.
        :param X: Input tensor
        :param parameters: mean, cov of the randomly initialized gaussian
        :return: Likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
        */
        int n_features = X.size(1);
        torch::Tensor mean = parameters["mean"];
        torch::Tensor cov = parameters["cov"];
        double determinant = torch::det(cov).data().item<double>();

        torch::Tensor likelihoods = torch::zeros(X.size(0), X.dtype());

        for(auto& i : range( static_cast<int>(X.size(0)), 0) ) {
        	torch::Tensor sample = X.index({i, Slice()});

            double coefficients = 1.0/ std::sqrt( std::pow((2.0 * M_PI), n_features*1.) * determinant);

            torch::Tensor dif = (sample - mean).reshape({1,-1});
            torch::Tensor exponent = torch::exp( -0.5 * torch::mm(torch::mm(dif, torch::pinverse(cov)), dif.t()));

            //likelihoods[i] = (coefficients * exponent);
            likelihoods.index_put_({torch::tensor({i}, torch::kLong)}, (coefficients * exponent));
        }

        return likelihoods;
    }

	torch::Tensor get_likelihood(torch::Tensor X) {
        /*
        Previously, we have initialized 3 different mean and covariance in random_gaussian_initialization(). Now around
        each of these mean and cov, we see likelihood of the each sample using multivariate gaussian distribution.
        :param X:
        :return: Storing the likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
        */
        int n_samples = X.size(0);
        torch::Tensor likelihoods_cls = torch::zeros({n_samples, k});

        for(auto& cls : range(k, 0)) {

            likelihoods_cls.index_put_({Slice(), cls}, multivariate_gaussian_distribution(X, parameters[cls]));
        }

        return likelihoods_cls;
	}

    void expectation(torch::Tensor X) {
        /*
        Expectation Maximization Algorithm is used to find the optimized value of randomly initialized mean and cov.
        Expectation refers to probability. Here, It calculates the probabilities of X belonging to different cluster.
        :param X: input tensor
        :return: Max probability of each sample belonging to a particular class.
        */
    	torch::Tensor weighted_likelihood = get_likelihood(X) * prior;

    	c10::OptionalArrayRef<long int> dim = {1};
    	torch::Tensor sum_likelihood =  torch::sum(weighted_likelihood, dim).unsqueeze(1);

        // Determine responsibility as P(X|y)*P(y)/P(X)
        // responsibility stores each sample's probability score corresponding to each class
        responsibility = weighted_likelihood / sum_likelihood;

		// Assign samples to cluster that has largest probability
        std::optional<long int> dm = {1};
        sample_assignments = responsibility.argmax(dm);

		// Save value for convergence check
        responsibilities.push_back(std::get<0>(torch::max(responsibility, 1)));
    }

    void maximization(torch::Tensor X) {
        /*
        Iterate through clusters and updating mean and covariance.
        Finding updated mean and covariance using probability score of each sample w.r.t each class
        :param X:
        :return: Updated mean, covariance and priors
        */

        for(auto& i : range(k, 0)) {
        	torch::Tensor resp = responsibility.index({Slice(), i}).unsqueeze(1);
        	c10::OptionalArrayRef<long int> dim={0};
        	torch::Tensor mean = torch::sum(resp * X, dim) / torch::sum(resp);
			torch::Tensor covariance = torch::mm((X - mean).t(), (X - mean) * resp) / resp.sum();

            //print('bf up cov: ', self.parameters[i]['cov'])
            (parameters[i])["mean"] = mean.unsqueeze(0);
            (parameters[i])["cov"]  =  covariance;
            //print('after up cov: ', self.parameters[i]['cov'])
        }
        int n_samples = X.size(0);
        c10::OptionalArrayRef<long int> dim={0};
        prior = responsibility.sum(dim) / n_samples;
    }

    bool convergence(torch::Tensor X, bool flag) {
        // Convergence if || likehood - last_likelihood || < tolerance
        if(responsibilities.size() < 2 )
            return false;

        torch::Tensor d = responsibilities[responsibilities.size() - 1] - responsibilities[responsibilities.size() - 2];
        double difference = torch::norm(d).data().item<double>();
        if(flag)
        	std::cout << "difference: " << difference << '\n';
        return difference <= tolerance;
    }

    torch::Tensor predict(torch::Tensor X, int step = 10) {
        random_gaussian_initialization(X);

        for(auto& _ : range(max_epochs, 0)) {
            expectation(X);
            maximization(X);

            bool flag = false;
            if( _ % step == 0)
            	flag = true;

            if(convergence(X, flag))
                break;
        }

        expectation(X);
        return sample_assignments;
    }


private:
	int k = 0, max_epochs = 10;
	double tolerance = 1e-5;
	torch::Tensor prior = torch::empty(0);
	std::vector<std::map<std::string, torch::Tensor>> parameters;
	torch::Tensor responsibility = torch::empty(0);
	std::vector<torch::Tensor> responsibilities{};
	torch::Tensor sample_assignments = torch::empty(0);
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::ifstream file;
	std::string path = "./data/iris.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records += 1; 		// if first row is not heads but record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

    // Create an unordered_map to hold label names
    std::unordered_map<std::string, int> iMap;
	iMap.insert({"Iris-setosa", 0});
	iMap.insert({"Iris-versicolor", 1});
	iMap.insert({"Iris-virginica", 2});

    std::cout << "iMap['benign']: " << iMap["benign"] << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeToensorIndex(num_records, true);

	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.80);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(sidx[i].data().item<int>());
	}

	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, false, false); // no normalization

	file.close();

	sidx = RangeToensorIndex(train_dt.size(0), true);
	train_dt = torch::index_select(train_dt, 0, sidx.squeeze());
	train_lab = torch::index_select(train_lab, 0, sidx.squeeze());

	sidx = RangeToensorIndex(test_dt.size(0), true);
	test_dt = torch::index_select(test_dt, 0, sidx.squeeze());
	test_lab = torch::index_select(test_lab, 0, sidx.squeeze());

	train_lab = train_lab.reshape({-1});
	test_lab = test_lab.reshape({-1});

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	std::cout <<  std::get<0>(at::_unique(train_lab)) << '\n';

	int n_classes = std::get<0>(at::_unique(train_lab)).size(0);
    GMM gmm(n_classes, 1000, 1e-7);

    //x_train = gmm.normalization(x_train)
    torch::Tensor y_pred = gmm.predict(train_dt, 50);
    std::cout << "y_pred dim: " << y_pred.dim() << '\n';
    std::cout << "train_lab: " << train_lab.sizes() << '\n';

    printf("Accuracy score: %3.2f %s\n", (accuracy(y_pred, train_lab)*100.0 / y_pred.size(0)), "%");

	std::cout << "Done!\n";
	return 0;
}





