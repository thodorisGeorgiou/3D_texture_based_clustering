#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include <cmath>
#include "mitra_var_2.h"

MITRA_VAR_2::MITRA_VAR_2(int n_threads, float thres){
	num_threads = n_threads;
	threshold = thres;
}

void MITRA_VAR_2::create_distance_histogramm(float *data, size_t data_step, int dims, int feature, long *histogramm, int num_threads){
	float pMax_distance[num_threads], pMin_distance[num_threads], max_distance = 0, min_distance = std::numeric_limits<float>::max();
	std::vector<std::thread> threads;
	// std::cout << "line 11" << std::endl;
	for(int t = 0; t < num_threads; t++){
		threads.push_back(std::thread(&MITRA_VAR_2::find_max_distance, this, t, num_threads, data, data_step, dims, feature, pMax_distance+t, pMin_distance+t));
	}
	for(auto& t: threads) t.join();
	// std::cout << "line 12" << std::endl;
	threads.clear();
	for(int t = 0; t<num_threads; t++){
		if(pMax_distance[t]>max_distance)
			max_distance = pMax_distance[t];
		if(pMin_distance[t]<min_distance)
			min_distance = pMin_distance[t];
	}
	float diff = max_distance - min_distance;
	// std::cout << max_distance << std::endl;
	// std::cout << min_distance << std::endl;
	// std::cout << diff << std::endl;
	// std::cout << "line 13" << std::endl;
	max_distance = std::abs(diff);
	// std::cout << max_distance << std::endl;
	// pHistogramm[100] = 1;
	// pHistogramm[200] = 2;
	// pHistogramm[300] = 3;
	// pHistogramm[0] = 0;
	// std::cout << "line 14" << std::endl;
	if(max_distance == 0){
		for(int b = 0; b<99; b++) histogramm[b] = 0;
		histogramm[99] = dims;
	}
	else{
		long *pHistogramm = new long[num_threads*100];
		for(int t = 0; t < num_threads; t++){
			threads.push_back(std::thread(&MITRA_VAR_2::thread_compute_histogramm, this, t, num_threads, data, data_step, dims, feature, pHistogramm+(100*t), max_distance));
		}
		for(auto& t: threads) t.join();
		// std::cout << "line 15" << std::endl;
		for(int t = 0; t<num_threads; t++){
			for(int b = 0; b<100; b++) histogramm[b] += pHistogramm[t*100 + b];
		}
		delete[] pHistogramm;
	}
}

int MITRA_VAR_2::serial_sum(int st, int end){
	int result = 0;
	for(int i=st; i<end; i++) result+=i;
	return result;
}

void MITRA_VAR_2::find_max_distance(int thread_id, int num_threads, float *data, size_t data_step, int dims, int feature, float *max_distance, float *min_distance){
	*max_distance = 0;
	*min_distance = std::numeric_limits<float>::max();
	for(int point_0 = thread_id; point_0<dims; point_0+=num_threads){
		// std::cout << data[point_0*data_step+feature] << std::endl;
		if(data[point_0*data_step+feature]>*max_distance)
			*max_distance = data[point_0*data_step+feature];
		if(data[point_0*data_step+feature]<*min_distance)
			*min_distance = data[point_0*data_step+feature];
	}
}

void MITRA_VAR_2::thread_compute_histogramm(int thread_id, int num_threads, float *data, size_t data_step, int dims, int feature, long *histogramm, float max_distance){
	// std::cout << histogramm[0] << std::endl;
	for(int b = 0; b<100; b++) histogramm[b] = 0;
	for(int point_0 = thread_id; point_0 < dims-1; point_0+=num_threads){
		// std::cout << point_0 << std::endl;
		// std::cout << p << std::endl;
		// std::cout << "-------" << std::endl;
		for(int point_1 = point_0+1; point_1<dims; point_1++){
			int distance = (int)((float)std::abs(data[point_0*data_step+feature] - data[point_1*data_step+feature])/max_distance * 100);
			switch(distance){
				case 100:
					histogramm[99]++;
					break;
				default:
					histogramm[distance]++;
					break;
			}
		}
	}
}

float MITRA_VAR_2::feature_entropy(cv::Mat _data, int feature, int num_threads){
	float *data = (float*)_data.data;
	size_t data_step = _data.step/sizeof(data[0]);
	long *histogramm = new long[100];
	int dims = _data.rows;
	long nPairs = ((long)dims*(long)(dims-1))/2;
	create_distance_histogramm(data, data_step, _data.rows, feature, histogramm,  num_threads);
	long q_min = (long)(((long double)0.005)*nPairs);
	int Ri = 20;
	int b;
	for(b = 0; b<100; b++)
		if(histogramm[b] > q_min) break;
	int maxBin = b;
	long maxBin_value = histogramm[b];
	for(b = maxBin+1; (b < maxBin+Ri)&&(b<100); b++){
		if(maxBin_value<histogramm[b]){
			maxBin = b;
			maxBin_value = histogramm[b];
		}
	}
	float mu = 0.1*log(50*expm1(((float)maxBin)*0.1)+1);
	long double entropy = 0;
	for(int b = 0; b<100; b++){
		float distance = 0.01*b;
		if(mu>distance)
			entropy += (long double)(histogramm[b]*(long double)(expm1(0.1*b)/expm1(10*mu)));
		else
			entropy += (long double)(histogramm[b]*(long double)(expm1(10 - b*0.1)/expm1(10*(1 - mu))));
	}
	delete[] histogramm;
	return (float)entropy;
}


void MITRA_VAR_2::checknDiscard(std::vector<MITRA_VAR_2::dist_struct> &entropy, int feature, cv::Mat &_data){
	float *data = (float*)_data.data;
	size_t step_data = _data.step/sizeof(data[0]);
	std::vector<std::thread> threads;
	float *distances = new float[entropy.size()-feature-1];
	void *src = (void*) &entropy;
	for(int t = 0; t< num_threads; t++){
		threads.push_back(std::thread(&MITRA_VAR_2::thread_distance_calculator, this, t, data, step_data, distances, (int)_data.rows, num_threads, feature, src));
	}
	for (auto& t: threads) t.join();
	for(int qfeature = entropy.size()-1; qfeature > feature ; qfeature--){
		if(distances[qfeature-feature-1] < threshold){
			// std::cout << qfeature << " " << entropy.size() << std::endl;
			entropy.erase(entropy.begin() + qfeature); 
		}
	}
	delete[] distances;
}

void MITRA_VAR_2::thread_distance_calculator(int thread_id, float *src, size_t step_src, float *distances, int dims, int calc_step, int cFeature, void *_entropy){
	std::vector<MITRA_VAR_2::dist_struct> &entropy = *((std::vector<MITRA_VAR_2::dist_struct> *)_entropy);
	int qFeature = entropy[cFeature].index;
	// std::cout << dims << std::endl;
	long double qFeature_length = 0;
	for(int d = 0; d<dims; d++){
		float val = src[qFeature + d*step_src];
		qFeature_length += (long double)(val*val);
	}
	qFeature_length = (long double)sqrt((double)(qFeature_length));
	for(int tfeature = cFeature+thread_id+1; tfeature < entropy.size(); tfeature+=calc_step){
		// std::cout << mean_feature << std::endl;
		// std::cout << var_feature << std::endl;
		int feature = entropy[tfeature].index;
		long double feature_length = 0;
		long double inner_prod = 0;
		for(int d = 0; d<dims; d++){
			register float qVal = src[qFeature + d*step_src];
			register float fVal = src[feature + d*step_src];
			feature_length += (long double)(fVal*fVal);
			inner_prod += (long double)(qVal*fVal);
		}
		feature_length = (long double)sqrt((double)(feature_length));
		float prod = (float)(feature_length*qFeature_length);
		if(prod == 0) distances[tfeature - cFeature - 1] = 1;
		else{
			// long double cov = 0;
			// for(int d = 0; d<dims; d++)
			// 	cov += ((long double)(src[feature + d*step_src]) - mean_feature)*((long double)(src[qFeature + d*step_src]) - mean_qFeature);
			// std::cout << cov << std::endl;
			// register float riza = (float)((var_qFeature+var_feature)*(var_qFeature+var_feature)-4*var_feature*var_qFeature*(1-cov*cov));
			distances[tfeature - cFeature - 1] = (float)(1-(inner_prod/prod));
		}
		// std::cout << distances[tfeature] << std::endl;
	}
		// std::cout << dst[feature].size() << std::endl;
}

void MITRA_VAR_2::sorted_insert(MITRA_VAR_2::dist_struct s, std::vector<MITRA_VAR_2::dist_struct> &vec){
	int temp = 0;
	register float distance = s.distance;
	while(temp < vec.size()){
		if(distance <= vec[temp].distance)
			break;
		temp++;
	}
	vec.insert(vec.begin()+temp, s);
}

int MITRA_VAR_2::select_features(cv::Mat &_data, cv::Mat &samples){
	std::vector<MITRA_VAR_2::dist_struct> entropy;
	std::cout << "line 1" << std::endl;
	for(int feature = 0; feature<samples.cols; feature++){
		MITRA_VAR_2::dist_struct s;
		s.distance = feature_entropy(samples, feature, num_threads);
		s.index = feature;
		sorted_insert(s, entropy);
		// std::cout << "feature: " << feature << " done!" << std::endl;
		// std::vector<MITRA_VAR_2::dist_struct> temp_vec;
		// sorted_distances.push_back(temp_vec);
	}
	std::cout << "line 2" << std::endl;
	for(int feature = 0; feature<entropy.size(); feature++){
		checknDiscard(entropy, feature, _data);
	}
	std::cout << "line 3" << std::endl;
	// std::cout << nn_features << std::endl;
	int nn_features = entropy.size();
	cv::Mat _new_features = cv::Mat(_data.rows, nn_features, CV_32F);
	float *data = (float*)_data.data;
	size_t step_d = _data.step/sizeof(data[0]);
	float *new_features = (float*)_new_features.data;
	size_t step_nf = _new_features.step/sizeof(new_features[0]);
	for(int feature = 0; feature < nn_features; feature++){
		for(int d = 0; d < _data.rows; d++){
			new_features[feature + d*step_nf] = data[entropy[feature].index+d*step_d];
			kept_features.push_back(entropy[feature].index);
		}
	}
	cv::swap(_data, _new_features);
	return nn_features;
}

MITRA_VAR_2::~MITRA_VAR_2(){
}