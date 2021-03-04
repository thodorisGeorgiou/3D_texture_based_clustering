#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "mitra.h"

MITRA::MITRA(int n_threads, int k){
	num_threads = n_threads;
	K = k;
}

void MITRA::distance_calculator(cv::Mat &_src, std::vector<std::vector<MITRA::dist_struct> > &dst){
	float *src = (float*)_src.data;
	size_t step_src = _src.step/sizeof(src[0]);
	std::vector<std::thread> threads;
	void *temp = (void*) &dst;
	for(int t = 0; t< num_threads; t++){
		threads.push_back(std::thread(&MITRA::thread_distance_calculator, this, t, src, step_src, temp, num_threads, _src.cols, _src.rows));
	}
	for (auto& t: threads) t.join();
	for(int feature = _src.cols - 1; feature >= 0 ; feature--) {
		// std::cout << dst[feature].size() << std::endl;
		// std::cout << "--------" << std::endl;
		for(int qFeature = 0; qFeature < dst[feature].size(); qFeature++){
			MITRA::dist_struct s;
			s.distance = dst[feature][qFeature].distance;
			s.index = feature;
			// std::cout << dst[feature][qFeature].index << std::endl;
			// std::cout << dst[feature][qFeature].distance << std::endl;
			sorted_insert(s, dst[dst[feature][qFeature].index]);
		}
	}
}

void MITRA::thread_distance_calculator(int thread_id, float *src, size_t step_src, void *_dst, int calc_step, int num_features, int dims){
	std::vector< std::vector<MITRA::dist_struct> > &dst = *((std::vector< std::vector<MITRA::dist_struct> > *)_dst);
	for(int feature = thread_id; feature < num_features - 1; feature+=calc_step){
		long double var_feature = 0;
		long double mean_feature = 0;
		for(int d = 0; d<dims; d++){
			float val = src[feature + d*step_src];
			var_feature += (long double)(val*val);
			mean_feature += (long double)val;
		}
		mean_feature = mean_feature/dims;
		var_feature = var_feature/dims - mean_feature*mean_feature;
		for(int qFeature = feature+1; qFeature<num_features; qFeature++){
			long double var_qFeature = 0;
			long double mean_qFeature = 0;
			long double cov = 0;
			for(int d = 0; d<dims; d++){
				register float val = src[qFeature + d*step_src];
				var_qFeature += (long double)(val*val);
				mean_qFeature += (long double)val;
				cov += (long double)(val*src[feature + d*step_src]);
			}
			mean_qFeature = mean_qFeature/dims;
			var_qFeature = var_qFeature/dims - mean_qFeature*mean_qFeature;
			MITRA::dist_struct s;
			s.index = qFeature;
			float prod = (float)(var_feature*var_qFeature);
			if(prod == 0) s.distance = 0;
			else{
				cov = (cov/dims - mean_qFeature*mean_feature)/sqrtf(prod);
				register float riza = (float)((var_qFeature+var_feature)*(var_qFeature+var_feature)-4*var_feature*var_qFeature*(1-cov*cov));
				s.distance = (float)(var_feature+var_qFeature-sqrtf(riza));
			}
			sorted_insert(s, dst[feature]);
		}
	}
}

void MITRA::sorted_insert(MITRA::dist_struct s, std::vector<MITRA::dist_struct> &vec){
	int temp = 0;
	register float distance = s.distance;
	while(temp < vec.size()){
		if(distance <= vec[temp].distance)
			break;
		temp++;
	}
	vec.insert(vec.begin()+temp, s);
}

int MITRA::find_mini(std::vector<std::vector<MITRA::dist_struct> > sorted_distances, bool *checked, bool *deleted){
	while(K>1){
		float minimum = std::numeric_limits<float>::max();
		int minimum_feature = -1;
		// std::cout << sorted_distances.size() << std::endl;
		for(int feature = 0; feature < sorted_distances.size(); feature++){
			int kapa = K;
			if((!checked[feature]) && !deleted[feature]){
				while(true){
					if(!deleted[sorted_distances[feature][kapa].index]){
						if(sorted_distances[feature][kapa].distance < minimum){
							minimum = sorted_distances[feature][kapa].distance;
							minimum_feature = feature;
						}
						break;
					}
					else{
						kapa++;
						if(kapa>=sorted_distances[feature].size()) break;
					}
				}
			}
			// std::cout << sorted_distances[feature].size() << std::endl;
		}
		if(minimum <= threshold){
			if(threshold == std::numeric_limits<float>::max()) threshold = minimum;
			// std::cout << minimum_feature << std::endl;
			return minimum_feature;
		}
		else
			K--;
	}
	return(-1);
}

int MITRA::select_features(cv::Mat &_data){
	std::vector<std::vector<MITRA::dist_struct> > sorted_distances;
	for(int feature = 0; feature<_data.cols; feature++){
		std::vector<MITRA::dist_struct> temp_vec;
		sorted_distances.push_back(temp_vec);
	}
	std::cout << "line 1" << std::endl;
	distance_calculator(_data, sorted_distances);
	std::cout << "line 2" << std::endl;
	bool checked[_data.cols], deleted[_data.cols];
	for(int i = 0; i<_data.cols; i++){
		checked[i] = false;
		deleted[i] = false;
	}
	threshold = std::numeric_limits<float>::max();
	int nn_features = _data.cols;
	while(true){
		int minimum_feature = find_mini(sorted_distances, checked, deleted);
		// std::cout << minimum_feature << std::endl;
		// std::cout << K << std::endl;
		if(minimum_feature == -1) break;
		checked[minimum_feature] = true;
		int kapa = 0, n_deleted = 0;
		while(true){
			// std::cout << sorted_distances[minimum_feature][kapa].index << std::endl;
			// std::cout << deleted[sorted_distances[minimum_feature][kapa].index] << std::endl;
			if(!deleted[sorted_distances[minimum_feature][kapa].index]){
				// std::cout << sorted_distances[minimum_feature][kapa].index << std::endl;
				deleted[sorted_distances[minimum_feature][kapa].index] = true;
				nn_features--;
				n_deleted++;
			}
			kapa++;
			if((kapa >= sorted_distances[minimum_feature].size()) || n_deleted > K - 1) break;
		}
		// std::cout << "------------" << std::endl;
	}
	std::cout << "line 3" << std::endl;
	// std::cout << nn_features << std::endl;
	sorted_distances.clear();
	cv::Mat _new_features = cv::Mat(_data.rows, nn_features, CV_32F);
	float *data = (float*)_data.data;
	size_t step_d = _data.step/sizeof(data[0]);
	float *new_features = (float*)_new_features.data;
	size_t step_nf = _new_features.step/sizeof(new_features[0]);
	int new_ind = 0;
	for(int feature = 0; feature < _data.cols; feature++){
		if(deleted[feature]) continue;
		for(int d = 0; d < _data.rows; d++) new_features[new_ind + d*step_nf] = data[feature+d*step_d];
		new_ind++;
	}
	cv::swap(_data, _new_features);
	return nn_features;
}

MITRA::~MITRA(){
}