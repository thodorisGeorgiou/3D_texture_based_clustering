#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "mitra_var_1.h"

MITRA_VAR_1::MITRA_VAR_1(int n_threads, float thres){
	num_threads = n_threads;
	threshold = thres;
}

void MITRA_VAR_1::distance_calculator(cv::Mat &_src, std::vector<std::vector<MITRA_VAR_1::dist_struct> > &dst){
	float *src = (float*)_src.data;
	size_t step_src = _src.step/sizeof(src[0]);
	std::vector<std::thread> threads;
	void *temp = (void*) &dst;
	dims = _src.rows;
	for(int t = 0; t< num_threads; t++){
		threads.push_back(std::thread(&MITRA_VAR_1::thread_mici_calculator, this, t, src, step_src, temp, _src.cols));
	}
	// for(int t = 0; t< num_threads; t++){
	// 	threads.push_back(std::thread(&MITRA_VAR_1::thread_cosine_calculator, this, t, src, step_src, temp, _src.cols));
	// }
	for (auto& t: threads) t.join();
	for(int feature = _src.cols - 1; feature >= 0 ; feature--) {
		for(int qFeature = 0; qFeature < dst[feature].size(); qFeature++){
			MITRA_VAR_1::dist_struct s;
			s.distance = dst[feature][qFeature].distance;
			s.index = feature;
			sorted_insert(s, dst[dst[feature][qFeature].index]);
		}
	}
}

void MITRA_VAR_1::thread_cosine_calculator(int thread_id, float *src, size_t step_src, void *_dst, int num_features){
	std::vector< std::vector<MITRA_VAR_1::dist_struct> > &dst = *((std::vector< std::vector<MITRA_VAR_1::dist_struct> > *)_dst);
	for(int feature = thread_id; feature < num_features - 1; feature+=num_threads){
		long double feature_length = 0;
		for(int d = 0; d<dims; d++){
			float val = src[feature + d*step_src];
			feature_length += (long double)(val*val);
		}
		feature_length = (long double)sqrt((double)(feature_length));
		for(int qFeature = feature+1; qFeature<num_features; qFeature++){
			long double qFeature_length = 0;
			long double inner_prod = 0;
			for(int d = 0; d<dims; d++){
				register float qVal = src[qFeature + d*step_src];
				register float fVal = src[feature + d*step_src];
				qFeature_length += (long double)(qVal*qVal);
				inner_prod += (long double)(qVal*fVal);
			}
			qFeature_length = (long double)sqrt((double)(qFeature_length));
			MITRA_VAR_1::dist_struct s;
			s.index = qFeature;
			float prod = (float)(feature_length*qFeature_length);
			if(prod == 0) s.distance = 1;
			else{
				s.distance = (float)(1-(inner_prod/prod));
			}
			sorted_insert(s, dst[feature]);
		}
	}
}

void MITRA_VAR_1::thread_mici_calculator(int thread_id, float *src, size_t step_src, void *_dst, int num_features){
	std::vector< std::vector<MITRA_VAR_1::dist_struct> > &dst = *((std::vector< std::vector<MITRA_VAR_1::dist_struct> > *)_dst);
	for(int feature = thread_id; feature < num_features - 1; feature+=num_threads){
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
			MITRA_VAR_1::dist_struct s;
			s.index = qFeature;
			float prod = (float)(var_feature*var_qFeature);
			if(prod == 0) s.distance = 1;
			else{
				cov = (cov/dims - mean_qFeature*mean_feature)/sqrtf(prod);
				register float riza = (float)((var_qFeature+var_feature)*(var_qFeature+var_feature)-4*prod*(1-cov*cov));
				s.distance = (float)(var_feature+var_qFeature-sqrtf(riza))/(float)(var_feature+var_qFeature);
				// s.distance = (float)(var_feature+var_qFeature-sqrtf(riza));
			}
			sorted_insert(s, dst[feature]);
		}
	}
}

void MITRA_VAR_1::sorted_insert(MITRA_VAR_1::dist_struct s, std::vector<MITRA_VAR_1::dist_struct> &vec){
	int temp = 0;
	register float distance = s.distance;
	while(temp < vec.size()){
		if(distance <= vec[temp].distance)
			break;
		temp++;
	}
	vec.insert(vec.begin()+temp, s);
}

int MITRA_VAR_1::find_mini(std::vector<std::vector<MITRA_VAR_1::dist_struct> > sorted_distances, bool *checked, bool *deleted){
	int minimum_feature = -1;
	int minimum_count = 0;
	// std::cout << sorted_distances.size() << std::endl;
	for(int feature = 0; feature < sorted_distances.size(); feature++){
		if(checked[feature] || deleted[feature]) continue;
		int current_count = 0;
		// std::cout << sorted_distances.size() << std::endl;
		for(int qFeature = 0; qFeature < sorted_distances[feature].size(); qFeature++){
			if(!deleted[sorted_distances[feature][qFeature].index]){
				// std::cout << sorted_distances[feature][qFeature].distance << std::endl;
				if(sorted_distances[feature][qFeature].distance < threshold)
					current_count++;
				else break;
			}
		}
		if(current_count > minimum_count){
			minimum_count = current_count;
			minimum_feature = feature;
		}
	}
	return(minimum_feature);
}

int MITRA_VAR_1::select_features(cv::Mat &_data){
	std::vector<std::vector<MITRA_VAR_1::dist_struct> > sorted_distances;
	for(int feature = 0; feature<_data.cols; feature++){
		std::vector<MITRA_VAR_1::dist_struct> temp_vec;
		sorted_distances.push_back(temp_vec);
	}
	// std::cout << "line 1" << std::endl;
	distance_calculator(_data, sorted_distances);
	// std::cout << "line 2" << std::endl;
	bool checked[_data.cols], deleted[_data.cols];
	for(int i = 0; i<_data.cols; i++){
		checked[i] = false;
		deleted[i] = false;
	}
	int nn_features = _data.cols;
	int counter = 1;
	while(true){
		int to_delete_feature = find_mini(sorted_distances, checked, deleted);
		// std::cout << "in this loop " << counter << " times" << std::endl;
		counter++;
		// std::cout << K << std::endl;
		if(to_delete_feature == -1) break;
		checked[to_delete_feature] = true;
		int kapa = 0;
		while(true){
			// std::cout << to_delete_feature << " " << sorted_distances[to_delete_feature][kapa].index << " " << 	sorted_distances[to_delete_feature][kapa].distance << std::endl;
			// std::cout << deleted[sorted_distances[to_delete_feature][kapa].index] << std::endl;
			// std::cout << deleted[sorted_distances[minimum_feature][kapa].index] << std::endl;
			if(!deleted[sorted_distances[to_delete_feature][kapa].index]){
				if(sorted_distances[to_delete_feature][kapa].distance < threshold){
				// std::cout << sorted_distances[minimum_feature][kapa].index << std::endl;
					// std::cout << sorted_distances[to_delete_feature][kapa].index << " " << 	sorted_distances[to_delete_feature][kapa].distance << std::endl;
					deleted[sorted_distances[to_delete_feature][kapa].index] = true;
					nn_features--;
				}
				else break;
			}
			kapa++;
			if((kapa >= sorted_distances[to_delete_feature].size())) break;
		}
	}
	// std::cout << "line 3" << std::endl;
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
		kept_features.push_back(feature);
		new_ind++;
	}
	cv::swap(_data, _new_features);
	return nn_features;
}

MITRA_VAR_1::~MITRA_VAR_1(){
}