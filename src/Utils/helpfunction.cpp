
#include "helpfunction.h"

LRdataset::LRdataset(std::pair<torch::Tensor, torch::Tensor> data_and_labels) {

    features_ = std::move(data_and_labels.first);
    labels_ = std::move(data_and_labels.second);
}

LRdataset::LRdataset(torch::Tensor data, torch::Tensor labels) {

    features_ = data;
    labels_ = labels;
}

torch::data::Example<> LRdataset::get(size_t index) {
    return {features_[index], labels_[index]};
}

torch::optional<size_t> LRdataset::size() const {
    return features_.size(0);
}

const torch::Tensor& LRdataset::features() const {
    return features_;
}

const torch::Tensor& LRdataset::labels() const {
    return labels_;
}

std::pair<torch::Tensor, torch::Tensor> synthetic_data(torch::Tensor true_w, float true_b, int64_t num_samples) {

	auto X = torch::normal(0.0, 1.0, {num_samples, true_w.size(0)});
	auto y = torch::matmul(X, true_w) + true_b;
	y += torch::normal(0.0, 0.01, y.sizes());
	y = torch::reshape(y, {-1, 1});

	return {X, y};
 }

// # Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
torch::Tensor linreg(torch::Tensor X, torch::Tensor w, torch::Tensor b) {
	// The linear regression model
	return torch::matmul(X, w) + b;
}

// # Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
torch::Tensor squared_loss(torch::Tensor y_hat, torch::Tensor y) {
	// Squared loss
	auto rlt = torch::pow(y_hat - torch::reshape(y, y_hat.sizes()), 2) / 2;
	return rlt;
}

void sgd(torch::Tensor& w, torch::Tensor& b, float lr, int64_t batch_size) {
	//Minibatch stochastic gradient descent.
	torch::NoGradGuard no_grad_guard;
	// SGD
	w -= (lr * w.grad() / batch_size);
	w.grad().zero_();

	b -= (lr * b.grad() / batch_size);
	b.grad().zero_();
}


torch::Tensor Softmax(torch::Tensor X) {
	torch::Tensor val_max, _;
	std::tie(val_max, _) = torch::max(X, -1, true);
	torch::Tensor X_exp = torch::exp(X - val_max);

	c10::OptionalArrayRef<long int> dim = {1};
	torch::Tensor partition = torch::sum(X_exp, dim, true);
	return (X_exp / partition);
}

torch::Tensor Sigmoid(torch::Tensor X) {

	if( torch::sum( (X < 0.).to(torch::kInt32) ).data().item<int>() > 0 ) {
		return torch::exp(X) / (1.0 + torch::exp(X));
	} else {
		return 1.0 / (1.0 + torch::exp(-1.0 * X));
	}
}

int64_t accuracy(torch::Tensor y_hat, torch::Tensor y) {
	if( y_hat.sizes().size() > 1 ) {
		if( y_hat.size(0) > 1 && y_hat.size(1) > 1 ) {
			std::optional<long int> dim = {1};
			y_hat = torch::argmax(y_hat, dim);
		}
	}

	y_hat = y_hat.to(y.dtype());

	auto cmp = (y_hat.squeeze() == y.squeeze() );

	return torch::sum(cmp.to(y.dtype())).item<int64_t>();

}

torch::Tensor l2_penalty(torch::Tensor x) {
	return (torch::sum(x.pow(2)) / 2);
}


// data batch indices
std::list<torch::Tensor> data_index_iter(int64_t num_examples, int64_t batch_size, bool shuffle) {

	std::list<torch::Tensor> batch_indices;
	// data index
	std::vector<int64_t> index;
	for (int64_t i = 0; i < num_examples; ++i) {
		index.push_back(i);
	}
	// shuffle index
	if( shuffle ) std::random_shuffle(index.begin(), index.end());

	for (int64_t i = 0; i < static_cast<int64_t>(index.size()); i +=batch_size) {
		std::vector<int64_t>::const_iterator first = index.begin() + i;
		std::vector<int64_t>::const_iterator last = index.begin() + std::min(i + batch_size, num_examples);
		std::vector<int64_t> indices(first, last);

		int64_t idx_size = indices.size();
		torch::Tensor idx = (torch::from_blob(indices.data(), {idx_size}, at::TensorOptions(torch::kInt64))).clone();

		//auto batch_x = X.index_select(0, idx);
		//auto batch_y = Y.index_select(0, idx);

		batch_indices.push_back(idx);
	}
	return( batch_indices );
}

torch::Tensor RangeTensorIndex(int64_t num, bool suffle) {
	std::vector<int64_t> idx;
	for( int64_t i = 0; i < num; i++ )
		idx.push_back(i);

	if( suffle ) {
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
	}

	torch::Tensor RngIdx = (torch::from_blob(idx.data(), {num}, at::TensorOptions(torch::kInt64))).clone();
	return RngIdx;
}

torch::Tensor vectorTotensor(std::vector<double>  x) {
	torch::Tensor tX = torch::from_blob(x.data(), {static_cast<int64_t>(x.size()), 1},
			at::TensorOptions(torch::kDouble)).clone();
	return tX;
}

std::vector<double> tensorTovector(torch::Tensor x) {
	vector<double> vX(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	return vX;
}


torch::Tensor polyf(torch::Tensor U, torch::Tensor beta) {
	auto t = torch::arange(0, beta.size(0)).to(U.dtype());
	std::cout << t.dtype() << '\n';
	auto m = torch::pow(U, t);
	//std::cout << m.dtype() << '\n';
	torch::Tensor tp = torch::matmul(m, beta.to(U.dtype()));
	//std::cout << tp.sizes() << '\n';
	return tp;
}

torch::Tensor polyfit(torch::Tensor U, torch::Tensor Y, int degree) {
	int num_dims = degree + 1;
	auto t = torch::arange(0, num_dims).to(U.dtype());
	auto M = torch::pow(U, t);
	std::cout << "M: " << M.sizes() << '\n';
	auto coeffs =  ((torch::inverse(M.t().mm(M))).mm(M.t())).mm(Y);
	std::cout << "coeffs: " << coeffs.dtype() << " " << coeffs.sizes() << '\n';
	return coeffs;
}

torch::Tensor to_categorical(torch::Tensor X, int n_col) {
    if( n_col < 1 )
        n_col = torch::amax(X).data().item<int>() + 1;
    //std::cout << "n_col: " << n_col << '\n';

    torch::Tensor one_hot = torch::zeros({X.size(0), n_col}, torch::kDouble);
    one_hot.index_put_({torch::arange(X.size(0)), X}, 1.);
    return one_hot;
}


std::string replace_all_char(std::string str, std::string replacement, std::vector<std::string> toBeReplaced) {
	for(auto& toberep : toBeReplaced) {
		//std::replace(str.begin(), str.end(), toberep, replacement);
	    size_t pos;
	    while ((pos = str.find(toberep)) != std::string::npos) {
	        str.replace(pos, 1, replacement);
	    }
	}
	return str;
}

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        if(!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

