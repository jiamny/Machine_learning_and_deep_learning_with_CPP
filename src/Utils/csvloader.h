#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>  	// for accumulate etc
#include <cmath>		// for sqrt
#include <unordered_map>
#include <unordered_set>
#include <torch/torch.h>
#include <regex>
using namespace std;


bool isNumberRegex(const std::string& str) {
	std::regex numberRegex("^[-+]?([0-9]*\\.[0-9]+[0-9]+)$");
	return std::regex_match(str, numberRegex);
}

class CSVRow {
public:
	float operator[](std::size_t index) {
		std::string &eg = m_data[index];
		return std::atof(eg.c_str());
	}

	std::string& operator()(std::size_t index) {
			std::string &eg = m_data[index];
			return eg;
	}

	int at(std::size_t index) {
		std::string &eg = m_data[index];
		return std::atoi(eg.c_str());
	}

	std::size_t size() const {
		return m_data.size();
	}
	void readNextRow(std::istream &str) {
		std::string line;
		std::getline(str, line);

		std::stringstream lineStream(line);
		std::string cell;

		m_data.clear();
		while (std::getline(lineStream, cell, ',')) {
			m_data.push_back(cell);
		}
		// This checks for a trailing comma with no data after it.
		if (!lineStream && cell.empty()) {
			// If there was a trailing comma then add an empty element.
			m_data.push_back("");
		}
	}
private:
	std::vector<std::string> m_data;
};

std::istream& operator>>(std::istream &str, CSVRow &data) {
	data.readNextRow(str);
	return str;
}

// Normalize Feature, Formula: (x - min)/(max - min)
std::vector<float> normalize_feature(std::vector<float> feat) {
	using ConstIter = std::vector<float>::const_iterator;
	ConstIter max_element; //= *std::max_element(feat.begin(), feat.end());
	ConstIter min_element; //= *std::min_element(feat.begin(), feat.end());
	std::tie(min_element, max_element) = std::minmax_element(std::begin(feat), std::end(feat));

	float max=*max_element, min=*min_element;

	float extra = max == min ? 1.0 : 0.0;
	std::vector<float> rlt;
	for(auto &val : feat) {
		// max_element - min_element + 1 to avoid divide by zero error
		rlt.push_back((val - min) / (max - min + extra));
	}

	return rlt;
}

// Normalize , Formula: (x - mean)/(stdev)
std::vector<float> normalize_dt(std::vector<float> feat) {

	float sum = std::accumulate(std::begin(feat), std::end(feat), 0.0);
	float m =  sum / feat.size();

	double accum = 0.0;
	std::for_each (std::begin(feat), std::end(feat), [&](const float d) {
	    accum += (d - m) * (d - m);
	});

	double stdev = sqrt(accum / (feat.size()-1));

	std::vector<float> rlt;
	for(auto &val : feat) {
		rlt.push_back((val - m) / stdev );
	}

	return rlt;
}

// This function processes data, Loads CSV file to vectors and normalizes
// Assumes last column to be label and first row to be header (or name of the features)
std::pair<std::vector<float>, std::vector<float>> process_data(
		std::ifstream &file, bool normalize_lable=false, bool zscore = false) {
	std::vector<std::vector<float>> features;
	std::vector<float> label;

	CSVRow row;
	// Read and throw away the first row.
	file >> row;

	int64_t n = 0;
	// last column is label
	while (file >> row) {
		if( n == 0 ) {
			for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
				std::vector<float> v;
				v.push_back(row[loop]*1.0);
				features.push_back(v);
			}

		} else {
			for( std::size_t loop = 0; loop < (row.size() - 1); ++loop )
				features[loop].push_back(row[loop]*1.0);
		}

		// Push final column to label vector
		label.push_back(row[row.size() - 1]);
		n++;
	}

	for( int b = 0; b < features.size(); b++ ) {
		if( zscore ) {
			features[b] = normalize_dt(features[b]);
		} else {
			features[b] = normalize_feature(features[b]);
		}
	}

	if( normalize_lable ) {
		if( zscore ) {
			label = normalize_dt(label);
		} else {
			label = normalize_feature(label);
		}
	}

	// Flatten features vectors to 1D
	std::vector<float> inputs;
	for (std::size_t i = 0; i < features[0].size(); i++) {
		for( int c = 0; c < features.size(); c++ )
			inputs.push_back((features[c])[i]);
	}

	return std::make_pair(inputs, label);
}

std::pair<std::vector<std::string>, std::vector<std::string>> process_bbc_news_data(
		std::ifstream &file, bool skip_header=true) {
	std::vector<std::string> texts;
	std::vector<std::string> topics;

	CSVRow row;
	// Read and throw away the first row.
	if(skip_header)
		file >> row;

	while (file >> row) {
		topics.push_back(row(0));
		texts.push_back(row(1));
	}

	return std::make_pair(topics, texts);
}


// This function processes data, Loads CSV file to vectors and normalizes features to (0, 1)
// Assumes last column to be label and first row to be header (or name of the features)
void process_split_data(std::ifstream &file, std::unordered_set<int> train_idx,
		std::vector<float> &train, std::vector<float> &trainLabel, std::vector<float> &test,
		std::vector<float> &testLabel, bool normalize_lable=false, bool zscore = false) {

	std::vector<std::vector<float>> trainData;
	std::vector<std::vector<float>> testData;

	CSVRow row;
	// Read and throw away the first row.
	file >> row;

	int j = 0;

	while (file >> row) {

		if (train_idx.count(j)) {

			if( j == 0 ) {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					std::vector<float> v;
					v.push_back(row[loop]*1.0);
					trainData.push_back(v);

					std::vector<float> m;
					testData.push_back(m);
				}

			} else {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop )
					trainData[loop].push_back(row[loop]*1.0);
			}

			// Push final column to label vector
			trainLabel.push_back(row[row.size() - 1]);
		} else {
			if( j == 0 ) {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					std::vector<float> v;
					v.push_back(row[loop]*1.0);
					testData.push_back(v);

					std::vector<float> m;
					trainData.push_back(m);
				}
			} else {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop )
					testData[loop].push_back(row[loop]*1.0);
			}

			// Push final column to label vector
			testLabel.push_back(row[row.size() - 1]);
		}
		j++;
	}

	for( int b = 0; b < trainData.size(); b++ ) {
		if( zscore ) {
			trainData[b] = normalize_dt(trainData[b]);
		} else {
			trainData[b] = normalize_feature(trainData[b]);
		}

	}

	for( int b = 0; b < testData.size(); b++ ) {
		if( zscore ) {
			testData[b] = normalize_dt(testData[b]);
		} else {
			testData[b] = normalize_feature(testData[b]);
		}

	}

	// Flatten features vectors to 1D
	train.clear();
	for (std::size_t i = 0; i < trainData[0].size(); i++) {
		for( int c = 0; c < trainData.size(); c++ )
			train.push_back((trainData[c])[i]);
	}

	test.clear();
	for (std::size_t i = 0; i < testData[0].size(); i++) {
		for( int c = 0; c < testData.size(); c++ )
			test.push_back((testData[c])[i]);
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> process_split_data2(
		std::ifstream &file, std::unordered_set<int> train_idx, std::unordered_map<std::string, int> iMap,
		bool skip_first_row = true, bool normalize_lable=false, bool zscore = false, bool normalize_data=true) {

	std::vector<std::vector<float>> trainData;
	std::vector<std::vector<float>> testData;
	std::vector<int> trainLabel, testLabel;

	CSVRow row;
	// Read and throw away the first row.
	if( skip_first_row )
		file >> row;

	int j = 0;

	while (file >> row) {

		if (train_idx.count(j)) {

			if( j == 0 ) {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					std::vector<float> v;
					v.push_back(row[loop]*1.0);
					trainData.push_back(v);

					std::vector<float> m;
					testData.push_back(m);
				}

			} else {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					trainData[loop].push_back(row[loop]*1.0);
				}
			}

			// Push final column to label vector
			int cls = row[row.size() - 1];
			if( ! iMap.empty() ) {
				cls = iMap[row(row.size() - 1)];
			}
			trainLabel.push_back(cls);

		} else {
			if( j == 0 ) {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					std::vector<float> v;
					v.push_back(row[loop]*1.0);
					testData.push_back(v);

					std::vector<float> m;
					trainData.push_back(m);
				}
			} else {
				for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					testData[loop].push_back(row[loop]*1.0);
				}
			}

			// Push final column to label vector
			int cls = row[row.size() - 1];
			if( ! iMap.empty() )
				cls = iMap[row(row.size() - 1)];
			testLabel.push_back(cls);
		}
		j++;
	}

	if( normalize_data ) {
		for( int b = 0; b < trainData.size(); b++ ) {
			if( zscore ) {
				trainData[b] = normalize_dt(trainData[b]);
			} else {
				trainData[b] = normalize_feature(trainData[b]);
			}
		}
	}

	int r = trainData[0].size();
	int c = trainData.size();
	std::vector<float> trData;
	for( int j = 0; j < r; j++ ) {
		for( int i = 0; i < c; i++ ) {
			std::vector<float> b = trainData[i];
			trData.push_back(b[j]);
		}
	}
	torch::Tensor train = torch::from_blob(trData.data(), {r, c}).clone();

	if( normalize_data ) {
		for( int b = 0; b < testData.size(); b++ ) {
			if( zscore ) {
				testData[b] = normalize_dt(testData[b]);
			} else {
				testData[b] = normalize_feature(testData[b]);
			}
		}
	}

	r = testData[0].size();
	c = testData.size();
	std::vector<float> tsData;
	for( int j = 0; j < r; j++ ) {
		for( int i = 0; i < c; i++ ) {
			std::vector<float> b = testData[i];
			tsData.push_back(b[j]);
		}
	}
	torch::Tensor test = torch::from_blob(tsData.data(), {r, c}).clone();;

	torch::Tensor train_label = torch::from_blob(trainLabel.data(),
			{int(trainLabel.size()), 1}, at::TensorOptions(torch::kInt)).clone();
	torch::Tensor test_label = torch::from_blob(testLabel.data(),
			{int(testLabel.size()), 1}, at::TensorOptions(torch::kInt)).clone();

	return std::make_tuple(train, train_label, test, test_label);
}

std::tuple<torch::Tensor, torch::Tensor> process_data2(
		std::ifstream &file, std::unordered_map<std::string, int> iMap,
		bool skip_first_row = true, bool normalize = false, bool zscore = false) {

	std::vector<std::vector<float>> tData;
	std::vector<int> tLabel;

	CSVRow row;
	// Read and throw away the first row.
	if( skip_first_row )
		file >> row;

	int j = 0;
	while (file >> row) {

		if( j == 0 ) {
			for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
				std::vector<float> v;
				v.push_back(row[loop]*1.0);
				tData.push_back(v);
			}

		} else {
			for( std::size_t loop = 0; loop < (row.size() - 1); ++loop ) {
					tData[loop].push_back(row[loop]*1.0);
			}
		}

		// Push final column to label vector
		int cls = iMap[row(row.size() - 1)];
		tLabel.push_back(cls);

		j++;
	}

	if( normalize ) {
		for( int b = 0; b < tData.size(); b++ ) {
			if( zscore ) {
				tData[b] = normalize_dt(tData[b]);
			} else {
				tData[b] = normalize_feature(tData[b]);
			}

		}
	}


	int r = tData[0].size();
	int c = tData.size();
	std::vector<float> trData;
	for( int j = 0; j < r; j++ ) {
		for( int i = 0; i < c; i++ ) {
			std::vector<float> b = tData[i];
			trData.push_back(b[j]);
		}
	}

	torch::Tensor train = torch::from_blob(trData.data(), {r, c}).clone();

	torch::Tensor train_label = torch::from_blob(tLabel.data(),
			{static_cast<long int>(tLabel.size()), 1}, at::TensorOptions(torch::kInt)).clone();

	return std::make_tuple(train, train_label);
}

std::tuple<torch::Tensor, torch::Tensor> load_label_fst_data(
		std::ifstream &file , bool dicimal = true,  bool normalize = false, int num_records = 0) {

	torch::Tensor data, target;
	std::vector<int> tLabel;

	if( dicimal ) {
		std::vector<std::vector<float>> tData;
		CSVRow row;
		int j = 0;

		while (file >> row) {
			// Push first column to label vector
			tLabel.push_back(row.at(0));

			if( j == 0 ) {
				for( std::size_t loop = 1; loop < row.size(); ++loop ) {
					std::vector<float> v;
					v.push_back(row[loop]*1.0);
					tData.push_back(v);
				}

			} else {
				for( std::size_t loop = 1; loop < row.size(); ++loop ) {
					tData[loop - 1].push_back(row[loop]*1.0);
				}
			}

			j++;
			if( num_records > 0 ) {
				if( j > num_records )
					break;
			}
		}

		if( normalize ) {
			for( int b = 0; b < tData.size(); b++ ) {
				tData[b] = normalize_feature(tData[b]);
			}
		}

		int r = tData[0].size();
		int c = tData.size();
		std::vector<float> trData;
		for( int j = 0; j < r; j++ ) {
			for( int i = 0; i < c; i++ ) {
				std::vector<float> b = tData[i];
				trData.push_back(b[j]);
			}
		}

		data = torch::from_blob(trData.data(), {r, c}, at::TensorOptions(torch::kFloat32)).clone();

	} else {
		std::vector<std::vector<int>> tData;
		CSVRow row;
		int j = 0;

		while (file >> row) {
			// Push first column to label vector
			tLabel.push_back(row.at(0));

			if( j == 0 ) {
				for( std::size_t loop = 1; loop < row.size(); ++loop ) {
					std::vector<int> v;
					v.push_back(row.at(loop));
					tData.push_back(v);
				}

			} else {
				for( std::size_t loop = 1; loop < row.size(); ++loop ) {
					tData[loop - 1].push_back(row.at(loop));
				}
			}

			j++;
			if( num_records > 0 ) {
				if( j > num_records )
					break;
			}
		}

		int r = tData[0].size();
		int c = tData.size();

		std::vector<int> trData;
		for( int j = 0; j < r; j++ ) {
			for( int i = 0; i < c; i++ ) {
				std::vector<int> b = tData[i];
				trData.push_back(b[j]);
			}
		}
		data = torch::from_blob(trData.data(), {r, c}, at::TensorOptions(torch::kInt)).clone();
	}

	target = torch::from_blob(tLabel.data(),
			{static_cast<long int>(tLabel.size()), 1}, at::TensorOptions(torch::kInt)).clone();

	return std::make_tuple(data, target);
}
