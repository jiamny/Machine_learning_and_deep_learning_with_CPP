
#ifndef SRC_UTILS_TEMPHELPFUNCTIONS_HPP_
#define SRC_UTILS_TEMPHELPFUNCTIONS_HPP_
#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <limits.h>
#include <cstdlib>

/*
 * There is no way to change the precision via to_string() but the setprecision IO manipulator could be used instead:
 */
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


template<typename T>
std::vector<T> range(const T count, T start = 1) {
    std::vector<T> aVector(count);
    iota(aVector.begin(), aVector.end(), start);
    return aVector;
}

template<typename T>
void printVector(std::vector<T> data) {

	std::cout << "[ ";
    #if __cplusplus==201103L
		std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
    #else
		std::for_each(data.begin(), data.end(), [](T v) {std::cout << v << ", ";});
	#endif
	std::cout << "]\n";
}

template<typename T>
std::vector<T> truncate_pad(std::vector<T> line, size_t num_steps, T padding_token) {
    //Truncate or pad sequences."""
    if( line.size() > num_steps ) {
    	std::vector<T> tokens(&line[0], &line[num_steps]);
        return tokens;  // Truncate
    } else {
    	int num_pad = num_steps - line.size();
    	for( int i = 0; i < num_pad; i++ )
    		line.push_back(padding_token);
    	return line;
    }
}

template<typename T>
T vector_sum(std::vector<T> data) {
    T total = 0;
    if( data.size() > 0 ) {
    	total = std::accumulate(data.begin(), data.end(), 0);
    }
    return total;
}

template<typename T>
inline size_t argMin(std::vector<T> a) {
    T min = 9999;
    size_t idx = 0;
    for( size_t i = 0; i < a.size(); i++ ) {
    	if( ! std::isnan(a[i]) ) {
    		if( a[i] < min ) {
    			idx = i;
    			min = a[i];
    		}
    	}
    }
	return idx;
}

template<typename T>
inline size_t argMax(std::vector<T> a) {
    T max = -9999;
    size_t idx = 0;
    for( size_t i = 0; i < a.size(); i++ ) {
    	if( ! std::isnan(a[i]) ) {
    		if( a[i] > max ) {
    			idx = i;
    			max = a[i];
    		}
    	}
    }
	return idx;
}

template<typename T>
std::vector<T> linspace(T start, T end, int length) {
	std::vector<T> vec;
	T diff = (end - start) / T(length);
	for (int i = 0; i < length; i++) {
		vec.push_back(start + diff * i);
	}
	return vec;
}

// This function adds two vectors and returns the sum vector
template<typename T>
std::vector<T> add_two_vectors(std::vector<T> const &a_vector,
		std::vector<T> const &b_vector) {
	// assert both are of same size
	assert(a_vector.size() == b_vector.size());

	std::vector<T> c_vector;
	std::transform(std::begin(a_vector), std::end(a_vector),
			std::begin(b_vector), std::back_inserter(c_vector),
			[](T const &a, T const &b) {
				return a + b;
			});

	return c_vector;
}

template<typename T>
T RandT(T _min, T _max) {
	T temp;
	if(_min > _max ) {
		temp = _min;
		_min = _max;
		_max = temp;
	}
	return( std::rand() / (double)RAND_MAX * (_max - _min) + _min );
}


template<typename T>
std::vector<T> random_choice(int const outputSize, const std::vector<T> samples, bool replacement = true,
								std::initializer_list<double> probabilities = {}) {

	std::vector<T> output;

	if(probabilities.size() > 0 ) {
		assert(samples.size() == probabilities.size());
		std::discrete_distribution<> distribution(probabilities);

	    std::vector<decltype(distribution)::result_type> indices;
	    indices.reserve(outputSize); // reserve to prevent reallocation
	    // use a generator lambda to draw random indices based on distribution
	    if( replacement ) {
		    std::generate_n(back_inserter(indices), outputSize,
		        [distribution = std::move(distribution), // could also capture by reference (&) or construct in the capture list
		         generator = std::default_random_engine{}  //pseudo random. Fixed seed! Always same output.
		        ]() mutable { // mutable required for generator
		            return distribution(generator);
		        });
	    } else {
        	auto generator = std::default_random_engine{};
        	indices.clear();
        	while( indices.size() < outputSize) {
            	auto d = distribution(generator);

            	std::vector<decltype(distribution)::result_type>::iterator it;
            	it = std::find(indices.begin(), indices.end(), d);
            	if(it == indices.end())
            		indices.push_back(d);
        	}
	    }

	    output.reserve(outputSize); // reserve to prevent reallocation
	    std::transform(cbegin(indices), cend(indices),
	        back_inserter(output),
	        [&samples](auto const index) {
	            return *std::next(cbegin(samples), index);
	            // note, for std::vector or std::array of samples, you can use
	            // return samples[index];
	        });

	} else {

		std::vector<int> indices;
		 if( replacement ) {
			 for(auto& _ : range(outputSize, 0)) {
				 int s_idx = RandT(0, static_cast<int>(samples.size() - 1));
				 indices.push_back(s_idx);
			 }
		 } else {

			 while(indices.size() < outputSize) {
				 int d = RandT(0, static_cast<int>(samples.size() - 1));
				 std::vector<int>::iterator it;
				 it = std::find(indices.begin(), indices.end(), d);
				 if(it == indices.end())
					 indices.push_back(d);
			 }
		 }

		 std::sort(indices.begin(), indices.end());

	     output.reserve(outputSize); // reserve to prevent reallocation
	     std::transform(cbegin(indices), cend(indices),
	          back_inserter(output),
	          [&samples](auto const index) {
	                return *std::next(cbegin(samples), index);
	                // note, for std::vector or std::array of samples, you can use
	                // return samples[index];
	     	 });
	}
    return output;
}


template<typename T>
T mostFrequent(std::vector<T> arr) {
    // code here
	int n = arr.size();
    int maxcount = 0;
    T element_having_max_freq;
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (arr[i] == arr[j])
                count++;
        }

        if(count > maxcount) {
            maxcount = count;
            element_having_max_freq = arr[i];
        }
    }

    return element_having_max_freq;
}

template<typename T>
std::string join(std::vector<T> const &vec, std::string delim) {
    if (vec.empty()) {
        return std::string();
    }

    std::stringstream ss;
    for (auto it = vec.begin(); it != vec.end(); it++)    {
        if (it != vec.begin()) {
            ss << delim;
        }
        ss << *it;
    }
    return ss.str();
}

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec)
{
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

template<typename V>
void combinations_with_replacement(V &v, size_t gp_sz, V &gp) {
    //V gp(gp_sz);
    auto total_n = std::pow(v.size(), gp.size());
    for (auto i = 0; i < total_n; ++i) {
        auto n = i;
        for (auto j = 0ul; j < gp.size(); ++j) {
            gp[gp.size() - j - 1] = v[n % v.size()];
            n /= v.size();
        }
    }
}

// Checks if s2 is a subset of s1.
template<typename T>
bool isSubset(std::vector<T> s1, std::vector<T> s2) {
	bool subSet = true;
    int i,j;
    int l1 = s1.size(), l2 = s2.size();
    for(i=0; i<l2; i++) {
        for(j=0; j<l1; j++) {
            if(s2[i]==s1[j])
                break;
        }
        if(j==l1)
        	return false;
    }
    return true;
}

// remove s1 elements in s2
template<typename T>
std::vector<T> subVector(std::vector<T> s1, std::vector<T> s2) {

	for(auto& elementToRemove : s2) {
	    // Remove the element using erase function and iterators
	    auto it = std::find(s1.begin(), s1.end(), elementToRemove);
	    // If element is found found, erase it
	    if (it != s1.end()) {
	        s1.erase(it);
	    }
	}
	return s1;
}

// common elements in s1 and s2
template<typename T>
std::vector<T> comVector(std::vector<T> s1, std::vector<T> s2) {
	std::vector<T> com;
	for(auto& element : s2) {
	    // Remove the element using erase function and iterators
	    auto it = std::find(s1.begin(), s1.end(), element);
	    // If element is found found, erase it
	    if (it != s1.end()) {
	        com.push_back(*it);
	    }
	}
	return com;
}

#endif /* SRC_UTILS_TEMPHELPFUNCTIONS_HPP_ */
