
#ifndef HELPFUNCTION_H_
#define HELPFUNCTION_H_

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

using namespace std;

class LRdataset : public torch::data::datasets::Dataset<LRdataset> {
 public:

    explicit LRdataset(std::pair<torch::Tensor, torch::Tensor> data_and_labels);
    explicit LRdataset(torch::Tensor data, torch::Tensor labels);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns all features.
    const torch::Tensor& features() const;

    // Returns all targets
    const torch::Tensor& labels() const;

 private:
    torch::Tensor features_;
    torch::Tensor labels_;
};

std::pair<torch::Tensor, torch::Tensor> synthetic_data(torch::Tensor w, float b, int64_t num_examples);

torch::Tensor linreg(torch::Tensor X, torch::Tensor w, torch::Tensor b);

torch::Tensor squared_loss(torch::Tensor y_hat, torch::Tensor y);

void sgd(torch::Tensor& w, torch::Tensor& b, float lr, int64_t batch_size);

torch::Tensor Softmax(torch::Tensor X);

torch::Tensor Sigmoid(torch::Tensor X);

torch::Tensor l2_penalty(torch::Tensor x);

int64_t accuracy(torch::Tensor y_hat, torch::Tensor y);

// data batch indices
std::list<torch::Tensor> data_index_iter(int64_t num_examples, int64_t batch_size, bool shuffle = true);

torch::Tensor RangeToensorIndex(int64_t num, bool suffle = false);

torch::Tensor vectorTotensor(std::vector<double>  x);

std::vector<double> tensorTovector(torch::Tensor x);

torch::Tensor polyfit(torch::Tensor U, torch::Tensor Y, int degree = 2);

torch::Tensor polyf(torch::Tensor U, torch::Tensor beta);

torch::Tensor to_categorical(torch::Tensor X, int n_col=0);

#endif /* HELPFUNCTION_H_ */