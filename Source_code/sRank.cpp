#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rank.h"
#include "sRank.h"


SRANK::SRANK(int n_threads, int d, float r, int s){
	num_threads = n_threads;
	dims = d;
	num_samples = s;
	ratio = r;
	OR = new long double[dims];
}

cv::Mat SRANK::random_samples(cv::Mat &data){
	int n_samples = (int)(data.rows*ratio);
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, (double)data.rows);
	cv::Mat samples = cv::Mat(n_samples, data.cols, CV_32F);
	bool picked[data.rows];
	for(int i = 0; i< data.rows; i++) picked[i] = false;
	for(int s = 0; s<n_samples; s++){
		int sample = (int)dist(mt);
		if(picked[sample]){
			s--;
			continue;
		}
		picked[sample] = true;
		for(int f = 0; f<data.cols; f++) samples.at<float>(s, f) = data.at<float>(sample, f);
	}
	return samples;
}

void SRANK::getRankedFeatures(cv::Mat &_data, std::vector<SRANK::rankedFeature> &rankedList){
	RANK rank(num_threads, dims);
	std::cout << "ratio = " << ratio << std::endl;
	std::fill_n(OR, dims, 0);
	for(int s = 0; s<num_samples; s++){
		std::cout << "\rs = " << s << std::flush;
		cv::Mat samples = random_samples(_data);
		float *data = (float *)samples.data;
		size_t dstep = samples.step/sizeof(data[0]);
		std::vector<RANK::rankedFeature> partial_rankings;
		rank.getRankedFeatures(data, dstep, samples.rows, partial_rankings);
		for(int f = 0; f<dims; f++){
			OR[partial_rankings[f].index] += partial_rankings[f].value;
		}
		partial_rankings.clear();
	}
	std::cout << std::endl;
	for(int f = 0; f<dims; f++){
		SRANK::rankedFeature ranking;
		ranking.index = f;
		ranking.value = OR[f];
		sorted_insert(rankedList, ranking);
	}
	std::cout << "End of sRank" << std::endl;
}

void SRANK::sorted_insert(std::vector<SRANK::rankedFeature> &vec, SRANK::rankedFeature s){
	int temp = 0;
	register float value = s.value;
	while(temp < vec.size()){
		if(value > vec[temp].value)
			break;
		temp++;
	}
	vec.insert(vec.begin()+temp, s);
}

SRANK::~SRANK(){
	delete[] OR;
}