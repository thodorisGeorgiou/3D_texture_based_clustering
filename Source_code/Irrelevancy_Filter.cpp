#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include <cmath>
#include "sRank.h"
#include "Irrelevancy_Filter.h"

IFilter::IFilter(int n_threads, float beta, int d, float r, int s){
	num_threads = n_threads;
	vita = beta;
	dims = d;
	sampling_ratio = r;
	num_samples = s;
	weights = new float[dims];
	std::fill_n(weights, dims, 1);
}

void IFilter::select_features(cv::Mat &_data){
	// std::vector<IFilter::fei_struct> feis;
	float *data = (float *)_data.data;
	size_t d_step = _data.step/sizeof(data[0]);
	num_points = _data.rows;
	std::vector<SRANK::rankedFeature> rankedFeatures;
	SRANK sRank(num_threads, dims, sampling_ratio, num_samples);
	sRank.getRankedFeatures(_data, rankedFeatures);
	calculate_weights(rankedFeatures);
}


void IFilter::calculate_D(float *data, size_t d_step){
		float Max_distance = 0;
		std::vector<std::thread> threads;
		float *pMax_distance = new float[num_threads];
		for(int t = 0; t<num_threads; t++){
			threads.push_back(std::thread(&IFilter::part_max_dist, this, t, data, d_step, pMax_distance+t));
		}
		for (auto& t: threads) t.join();
		threads.clear();
		for(int t=0; t<num_threads; t++)
			if(pMax_distance[t]>Max_distance)
				Max_distance = pMax_distance[t];
		D = vita*Max_distance;
		delete[] pMax_distance;
}

void IFilter::calculate_weights(std::vector<SRANK::rankedFeature> vec){
	long double avg = 0;
	for(int feature = 0; feature<dims-1; feature++){
		avg += vec[feature].value - vec[feature+1].value;
	}
	avg = avg/(dims-1);
	long double baseline = vec[dims-1].value - avg;
	long double DH[dims];
	for(int feature = 0; feature<dims; feature++){
		DH[feature] = vec[feature].value - baseline;
	}
	for(int feature = 0; feature<dims; feature++) weights[vec[feature].index] =  (float)(DH[feature] / DH[0]);
}

void IFilter::part_max_dist(int thread_id, float *data, size_t d_step, float *max_distance){
	*max_distance = 0;
	for(int p1 = thread_id; p1<num_points-1; p1+=num_threads){
		float *point1 = data+p1*d_step;
		for(int p2 = p1+1; p2 < num_points; p2++){
			float *point2 = data+p2*d_step;
			float diff = 0;
			for(int d = 0; d<dims; d++) diff += (float)pow(weights[d]*(point1[d] - point2[d]), 2);
			diff = sqrtf(diff);
			if (diff > *max_distance) *max_distance = diff;
		}
	}
}

float IFilter::calculate_fei(float *data, size_t d_step){
	double fei = 0;
	double par_fei[num_threads];
	std::vector<std::thread> threads;
	for(int t = 0; t<num_threads; t++){
		par_fei[t] = 0;
		threads.push_back(std::thread(&IFilter::thread_calculate_fei, this, t, data, d_step, par_fei+t));
	}
	for (auto& t: threads) t.join();
	threads.clear();
	for(int t = 0; t<num_threads; t++){
		fei += par_fei[t];
	}
	return (float)fei;
}

void IFilter::thread_calculate_fei(int thread_id, float *data, size_t d_step, double *fei){
	for(int p1 = thread_id; p1<num_points - 1; p1 += num_threads){
		float *point1 = data+p1*d_step;
		for(int p2 = p1+1; p2 < num_points; p2++){
			float *point2 = data+p2*d_step;
			float mu_t = 0, mu_zero = 0;
			float diffs[dims];
			std::fill_n(diffs, dims, 0);
			for(int d = 0; d<dims; d++){
				diffs[d] = point1[d] - point2[d];
				diffs[d] = diffs[d]*diffs[d];
				mu_zero += diffs[d];
				mu_t += weights[d]*weights[d]*diffs[d];
			}
			mu_t = sqrtf(mu_t);
			mu_zero = sqrtf(mu_zero);
			if(mu_t>D) mu_t = 0;
			else mu_t = 1 - (mu_t/D);
			if(mu_zero>D) mu_zero = 0;
			else mu_zero = 1 - (mu_zero/D);
			*fei += mu_t*(1 - mu_zero) + mu_zero*(1 - mu_t);
		}
	}
	*fei = *fei/((unsigned long)num_points*((unsigned long)num_points-1));
}

void IFilter::transorm_data(cv::Mat &_data){
	float *data = (float *)_data.data;
	size_t step = _data.step/sizeof(data[0]);
	for(int i = 0; i<_data.rows; i++){
		float *point = step*i + data;
		for(int j = 0; j<_data.cols; j++){
			point[j] = point[j]*weights[j];
		}
	}
}

float *IFilter::getWeights(){
	float *temp = new float[dims];
	for(int d = 0; d<dims; d++) temp[d] = weights[d];
	return temp;
}

IFilter::~IFilter(){
	delete[] weights;
}