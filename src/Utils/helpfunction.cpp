
#include "helpfunction.h"

using torch::indexing::Slice;
using torch::indexing::None;

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
    	item = strip(item);
        if(! item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

std::vector<std::vector<double>> get_mnist_image(torch::Tensor image) {
	int ncols = 28, nrows = 28;
	std::vector<std::vector<double>> C;
	for( int i = 0; i < nrows; i++ ) {
		std::vector<double> c;
		for( int j = 0; j < ncols; j++ )
			c.push_back(image[i][j].item<double>());
		C.push_back(c);
	}
	return C;
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor> generate_sequences(int n, bool variable_len, int seed) {
    torch::Tensor basic_corners = torch::tensor({{-1, -1}, {-1, 1}, {1, 1}, {1, -1}});
    torch::manual_seed(seed);

    torch::Tensor bases = torch::randint(4, n);
    torch::Tensor lengths;
    if( variable_len ) {
        lengths = torch::randint(3, n) + 2;
    } else {
        lengths = torch::ones({n})*4;
    }
    torch::Tensor directions = torch::randint(2, n);

    std::vector<torch::Tensor> points;
    for(auto& j : range(n, 0)) {
    	int b = bases[j].data().item<int>();
    	int d = directions[j].data().item<int>();
    	int l = lengths[j].data().item<int>();
    	torch::Tensor idx = torch::zeros({4}, torch::kInt32);
    	for(auto& i : range(4, 0)) {
    		idx[i] = (b + i ) % 4;
    	}
    	torch::Tensor pts = basic_corners.index_select(0, idx).to(torch::kFloat32);
    	if( d*2 - 1 < 0 )
    		pts = pts.flip({0});
    	pts = pts.index({Slice(0, l), Slice()}) + torch::randn({l, 2})*0.1;
    	points.push_back(pts);
    }

    //points = [basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)][:l] + np.random.randn(l, 2) * 0.1 for b, d, l in zip(bases, directions, lengths)]
    return std::make_tuple(points, directions);
}

std::pair<torch::nn::Linear, torch::nn::Linear> linear_layers(torch::Tensor Wx, torch::Tensor bx,
																torch::Tensor Wh, torch::Tensor bh) {
    int hidden_dim = Wx.size(0), n_features = Wx.size(1);
    torch::nn::Linear lin_input = torch::nn::Linear(torch::nn::LinearOptions(n_features, hidden_dim));
    torch::nn::Linear lin_hidden = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim));
    {
    	torch::NoGradGuard no_grad;
    	lin_input->weight = Wx;
    	lin_input->bias = bx;

    	lin_hidden->weight = Wh;
    	lin_hidden->bias = bh;
    }

    return std::make_pair(lin_hidden, lin_input);
}

// data batch indices
std::list<std::vector<int>> data_index_iter(int num_examples, int batch_size, bool shuffle) {

	std::list<std::vector<int>> batch_indices;
	// data index
	std::vector<int> index;
	for (int64_t i = 0; i < num_examples; ++i) {
		index.push_back(i);
	}
	// shuffle index
	if( shuffle ) std::random_shuffle(index.begin(), index.end());

	for (int i = 0; i < index.size(); i +=batch_size) {
		std::vector<int>::const_iterator first = index.begin() + i;
		std::vector<int>::const_iterator last = index.begin() + std::min(i + batch_size, num_examples);
		std::vector<int> indices(first, last);

		batch_indices.push_back(indices);
	}
	return( batch_indices );
}

std::string strip( const std::string& s ) {
	const std::string WHITESPACE = " \n\r\t\f\v";

	size_t start = s.find_first_not_of(WHITESPACE);
	std::string ls = (start == std::string::npos) ? "" : s.substr(start);

	size_t end = ls.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : ls.substr(0, end + 1);
}



