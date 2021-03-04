#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <thread>
#include <cmath>
#include "FuzzyCMeans.h"

FuzzyCMeans::FuzzyCMeans(int num_thr, float thres, int c, int dimensions, int nPoints, int fuzzyness){
	num_threads = num_thr;
	threshold = thres;
	num_points = nPoints;
	C = c;
	dims = dimensions;
	m = fuzzyness;
	membership = new float*[num_points];
	for(int d = 0; d < num_points; d++){
		membership[d] = new float[C];
		std::fill_n(membership[d], C, 0);
	}
	par_Vals = new FuzzyCMeans::parVals[num_threads];
}

// float FuzzyCMeans::euclidianSqr(float *point1, float *point2, int dims){
// 	float dist = 0;
// 	for(int d = 0; d < dims; d++){
// 		register float diff = point1[d] - point2[d];
// 		dist += diff*diff;
// 	}
// 	return dist;
// }

void FuzzyCMeans::UpdateMembership(int thread_id, const float *_data, size_t step, float *centers, size_t stepc){
	float em = 1.0/(m - 1);
	for(int i = thread_id; i < num_points; i += num_threads){
		float dist[C], distSum = 0;
		const float *point = _data + step*i;
		for(int c=0; c<C; c++){
			dist[c] = (float)pow(cv::normL2Sqr_(point, centers + stepc*c, dims), em);
			distSum += 1.0/dist[c];
		}
		// distSum = 1.0/distSum;
		for(int c=0; c<C; c++)
			membership[i][c] = (float)distSum/(float)dist[c];
	}
}

void FuzzyCMeans::PartUpdateCenters(int thread_id, const float *data, size_t step){
	FuzzyCMeans::parVals dst = par_Vals[thread_id];
	float **centers = dst.centers;
	float *memberSum = dst.memberSum;
	std::fill_n(memberSum, C, 0.0);
	for(int c = 0; c<C; c++){
		float *center = centers[c];
		std::fill_n(center, dims, 0.0);
		for(int i = thread_id; i < num_points; i+=num_threads){
			float mem = pow(membership[i][c], m);
			memberSum[c] += mem;
			const float *point = data + step*i;
			for(int d = 0; d < dims; d++) center[d] += point[d]*mem;
		}
	}
}

void FuzzyCMeans::calcGlobalCenters(int thread_id, float *centers, size_t stepc){
	for(int c = thread_id; c < C; c += num_threads){
		float *center = centers + stepc*c;
		float mem = 0;
		std::fill_n(center, dims, 0.0);
		for(int t = 0; t < num_threads; t++){
			float *parCenter = par_Vals[t].centers[c];
			mem += par_Vals[t].memberSum[c];
			for(int d = 0; d < dims; d++) center[d] += parCenter[d];
		}
		for(int d = 0; d < dims; d++) center[d] = center[d]/mem;
	}
}

void FuzzyCMeans::UpdateCenters(float *centers, size_t stepc, const float *data, size_t step){
	std::vector<std::thread> threads;
	for(int t = 0; t < num_threads; t++){
		threads.push_back(std::thread(&FuzzyCMeans::PartUpdateCenters, this, t, data, step));
	}
	for(auto& t: threads) t.join();
	threads.clear();
	for(int t = 0; t < num_threads; t++)
		threads.push_back(std::thread(&FuzzyCMeans::calcGlobalCenters, this, t, centers, stepc));
	for(auto& t: threads) t.join();
}

void FuzzyCMeans::cluster(const cv::Mat &data, cv::Mat &centers){
	const float *_data = (float*)data.data;
	size_t step = data.step/sizeof(_data[0]);
	cv::Mat new_centers(C, dims, CV_32F);
	std::vector<std::thread> threads;
	for(int t = 0; t < num_threads; t++){
		par_Vals[t].centers = new float*[C];
		par_Vals[t].memberSum = new float[C];
		for(int c = 0; c < C; c++)
			par_Vals[t].centers[c] = new float[dims];
	}
	int itter = 0;
	bool stop = true;
	while(stop){
		std::cout << "\rItteration #" << itter << std::flush;	
		float *_centers = (float*)centers.data;
		size_t stepc = centers.step/sizeof(_centers[0]);
		float *_newCenters = (float*)new_centers.data;
		size_t stepnc = new_centers.step/sizeof(_newCenters[0]);
		for(int t = 0; t < num_threads; t++)
			threads.push_back(std::thread(&FuzzyCMeans::UpdateMembership, this, t, _data, step, _centers, stepc));
		for(auto& t: threads) t.join();
		threads.clear();
		UpdateCenters(_newCenters, stepnc, _data, step);
		stop = checkCriterion(_centers, stepc, _newCenters, stepnc);
		cv::swap(centers, new_centers);
		itter++;
	}
	std::cout << std::endl;
	for(int t = 0; t < num_threads; t++){
		for(int c = 0; c < C; c++) delete[] par_Vals[t].centers[c];
		delete[] par_Vals[t].centers;
		delete[] par_Vals[t].memberSum;
	}
}

bool FuzzyCMeans::checkCriterion(float *_centers, size_t stepc, float *_newCenters, size_t stepnc){
	for(int c = 0; c < C; c++){
		if(cv::normL2Sqr_(_centers + stepc*c, _newCenters + c*stepnc, dims) > threshold)
			return true;
	}
	return false;
}

cv::Mat FuzzyCMeans::getLabels(){
	cv::Mat _labels = cv::Mat(num_points, 1, CV_32S, cv::Scalar(0));
	int *labels = (int*)_labels.data;
	for(int i = 0; i < num_points; i++){
		float max = 0;
		float *point = membership[i];
		for(int c = 0; c<C; c++){
			if(point[c] > max){
				labels[i] = c;
				max = point[c];
			}
		}
	}
	return _labels;
}

FuzzyCMeans::~FuzzyCMeans(){
	delete[] par_Vals;
	for(int d = 0; d < num_points; d++)
		delete[] membership[d];
	delete[] membership;
}