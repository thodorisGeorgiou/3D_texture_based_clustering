#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include <cmath>
#include "clusterScatteringCriteria.h"

CSCritteria::CSCritteria(int n_threads){
	num_threads = n_threads;
}

float CSCritteria::calculateCSC(cv::Mat &_centers, cv::Mat &_data, cv::Mat &_labels){
	float *data = (float*)_data.data;
	size_t d_step = _data.step/sizeof(data[0]);
	total_num_points = _data.rows;
	float *centers = (float*)_centers.data;
	size_t c_step = _centers.step/sizeof(centers[0]);
	int *labels = (int *)_labels.data;
	num_centers = _centers.rows;
	dims = _centers.cols;
	unsigned long num_points[num_centers];
	float global_center[dims];
	std::fill_n(num_points, num_centers, 0);
	std::fill_n(global_center, dims, 0);
	std::vector<std::thread> threads;
	for(int point = 0; point<_labels.rows; point++) num_points[_labels.at<int>(point)]++;
	for(int c = 0; c<num_centers; c++){
		for(int d = 0; d<dims; d++) global_center[d] += num_points[c]*centers[d + c*c_step];
	}
	for(int d = 0; d<dims; d++) global_center[d] = global_center[d]/(double)total_num_points;
	p_b = cv::Mat(dims, dims, CV_32F, cv::Scalar(0));
	p_w = cv::Mat(dims, dims, CV_32F, cv::Scalar(0));
	for(int c = 0; c<num_centers; c++)
		multiplyNadd_vectors(centers+c_step*c, global_center, p_b);
	for(int t = 0; t<num_threads; t++){
		cv::Mat temp = cv::Mat(dims, dims, CV_32F, cv::Scalar(0));
		for(int c = 0; c<num_centers; c++) part_p_w.push_back(temp.clone());
		threads.push_back(std::thread(&CSCritteria::calculate_p_w, this, t, data, d_step, centers, c_step, labels));
	}
	for (auto& t: threads) t.join();
	for(int t = 0; t<num_threads; t++)
		cv::add(p_w, part_p_w[t], p_w);
	cv::Mat inverted;
	cv::invert(p_w, inverted, cv::DECOMP_SVD);
	cv::Scalar trace = cv::trace(inverted*p_b);
	return (float)trace[0];
	// return (float)trace;
}

void CSCritteria::multiplyNadd_vectors(float *vec_1, float *vec_2, cv::Mat &_dst){
	float *dst = (float *)_dst.data;
	size_t dst_step = _dst.step/sizeof(dst[0]);
	for(int i = 0; i<dims; i++)
		for(int j = 0; j<dims; j++)
			dst[j + i*dst_step] += vec_1[i]*vec_2[j];
}

void CSCritteria::calculate_p_w(int thread_id, float *data, size_t d_step, float *centers, size_t c_step, int *labels){
	cv::Mat pp_w = part_p_w[thread_id];
	// float *pp_w = (float *)part_p_w[thread_id].data;
	// size_t pw_step = part_p_w[thread_id].step/sizeof(pp_w[0]);
	for(int p = thread_id; p<total_num_points; p+= num_threads){
		multiplyNadd_vectors(data+p*d_step, centers+c_step*labels[p], pp_w);
	}
}

CSCritteria::~CSCritteria(){
}