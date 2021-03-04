#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "rlm3D.h"

#include <unistd.h>
RLM_3d::RLM_3d(int x_rad, int y_rad, int z_rad, int x_step, int y_step, int z_step, int val_range, int n_threads){
	x_radius = x_rad;
	y_radius = y_rad;
	z_radius = z_rad;
	step_x = x_step;
	step_y = y_step;
	step_z = z_step;
	value_range = val_range;
	num_threads = n_threads;
	t_limits = (RLM_3d::limit_str **)malloc(num_threads*sizeof(RLM_3d::limit_str *));
	// t_pixel_values = (cv::Mat **)malloc(num_threads*sizeof(cv::Mat *));
	for(int t = 0; t<num_threads; t++){
		t_limits[t] = (RLM_3d::limit_str*)malloc(sizeof(RLM_3d::limit_str));
	}
	// pixel_values = NULL;
}

void RLM_3d::pixel_RLM(unsigned char ***src, int pixel[3], cv::Mat &dst, int direction){
	int radee[] = {2*x_radius+1, 2*y_radius+1, 2*z_radius+1};
	int neighbor_i[] = {step_x, 0, 0, step_x, -step_x, step_x, step_x, step_x, 0, step_x, -step_x, 0};
	int neighbor_j[] = {0, step_y, 0, step_y, step_y, -step_y, step_y, 0, step_y, -step_y, 0, step_y};
	int neighbor_k[] = {0, 0, step_z, step_z, step_z, step_z, 0, step_z, step_z, 0, step_z, -step_z};
	std::vector<int> lengths;
	if(neighbor_i[direction] != 0)
		lengths.push_back(((radee[0] - 1)/step_x) + 1);
	if(neighbor_j[direction] != 0)
		lengths.push_back(((radee[1] - 1)/step_y) + 1);
	if(neighbor_k[direction] != 0)
		lengths.push_back(((radee[2] - 1)/step_z) + 1);
	int sizes[] = {value_range, minimum(lengths)};
	// std::cout << sizes << std::endl;
	dst = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	unsigned char map[2*x_radius+1][2*y_radius+1][2*z_radius+1];
	for(int k = 0; k<2*z_radius+1; k++){
		for(int j = 0; j<2*y_radius+1; j++){
			for(int i = 0; i<2*x_radius+1; i++){
				map[i][j][k] = 0;
			}
		}
	}
	for(int k=pixel[2]-z_radius; k<=pixel[2]+z_radius; k++){
		for(int i=pixel[0]-x_radius; i<=pixel[0]+x_radius; i++){
			for(int j=pixel[1]-y_radius; j<=pixel[1]+y_radius; j++){
				if(map[i-pixel[0]+x_radius][j-pixel[1]+y_radius][k-pixel[2]+z_radius] == 1) continue;
				int c_pix[3] = {i, j, k}, c_length = 1, color = (int)src[i][j][k];
				while(true){
					c_pix[0] += neighbor_i[direction];
					c_pix[1] += neighbor_j[direction];
					c_pix[2] += neighbor_k[direction];
					if((c_pix[0] > pixel[0]+x_radius) || (c_pix[1] > pixel[1]+y_radius) ||(c_pix[2] > pixel[2]+z_radius) || 
						(c_pix[0] < pixel[0]-x_radius) || (c_pix[1] < pixel[1]-y_radius) ||(c_pix[2] < pixel[2]-z_radius)) break;
					if((int)src[c_pix[0]][c_pix[1]][c_pix[2]] == color){
						c_length++;
						map[c_pix[0]-pixel[0]+x_radius][c_pix[1]-pixel[1]+y_radius][c_pix[2]-pixel[2]+z_radius] = 1;
						continue;
					}
					break;
				}
				dst.at<float>(color, c_length) += 1;
			}
		}
	}
}

void RLM_3d::downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]){
	unsigned char scale = 256/value_range;
	for(int i=0; i < dim[0]; i++){
		for(int j=0; j < dim[1]; j++){
			for(int k=0; k < dim[2]; k++){
				dst[i][j][k] = src.at<uchar>(i,j,k)/scale;
			}
		}
	}
}

void RLM_3d::computeFeatures(cv::Mat *_src, int dims[3], int *mask){
	std::vector<std::thread> threads;
	int dim[3] = {dims[0] + 2*x_radius, dims[1] + 2*y_radius, dims[2] + 2*z_radius};
	cv::Mat src = addBoarder(_src, dims);
	int maskCount = 0;
	for(int p = 0; p < dims[0]*dims[1]*dims[2]; p++)
		if(mask[p]) maskCount++;
	map = new int[maskCount];
	int maskC = 0;
	for(int p = 0; p < dims[0]*dims[1]*dims[2]; p++){
		if(mask[p]){
			map[maskC] = p;
			maskC++;
		}
	}
	unsigned char ***tmp_src;
	tmp_src = (unsigned char ***)malloc(dim[0]*sizeof(unsigned char **));
	for(int i = 0; i < dim[0]; i++){
		tmp_src[i] = (unsigned char **)malloc(dim[1]*sizeof(unsigned char *));
		for(int j = 0; j < dim[1]; j++){
			tmp_src[i][j] = (unsigned char *)malloc(dim[2]*sizeof(unsigned char));
		}
	}
	downsize_matrix(src, tmp_src, dim);
	chunks(maskCount);
	int sizes[] = {maskCount, 60};
	pixel_values = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	for(int t = 0; t<num_threads; t++){
		threads.push_back(std::thread(&RLM_3d::thread_computeFeatures, this, t, dims, tmp_src));
	}
	for (auto& t: threads) t.join();
	for(int i = 0; i < dim[0]; i++){
		for(int j = 0; j < dim[1]; j++){
			free(tmp_src[i][j]);
		}
		free(tmp_src[i]);
	}
	free(tmp_src);
	delete[] map;
}

void RLM_3d::thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src){
	// cv::Mat *feature_mat = t_pixel_values[thread_id];
	int start = t_limits[thread_id]->st;
	int end = t_limits[thread_id]->end;
	float *data = (float*)pixel_values.data;
	size_t d_step = pixel_values.step/sizeof(data[0]);
	for(int p = start; p < end; p++){
		float *feature_pos = data + p*d_step;
		int mod = map[p] % (dim[1]*dim[0]);
		int pixel[] = {mod % dim[0] + x_radius, mod / dim[0] + y_radius, map[p] / (dim[1]*dim[0]) + z_radius};
		for(int direction = 0; direction<12; direction++){
			cv::Mat rlm_matrix;
			pixel_RLM(src, pixel, rlm_matrix, direction);
			calculateFeatures(rlm_matrix, feature_pos);
			feature_pos += 5;
		}
	}
}

void RLM_3d::calculateFeatures(cv::Mat rlm_matrix, float *feature_pos){
	// RLM_3d::features_st features;
	// features.short_run_emphasis = 0;
	// features.long_run_emphasis = 0;
	// features.run_length_non_unformity = 0;
	// features.gray_level_non_unformity = 0;
	float temp_run_sum[rlm_matrix.cols];
	for(int i = 0; i < rlm_matrix.cols; i++) temp_run_sum[i] = 0;
	float sum = cv::sum(rlm_matrix)[0];
	// features.run_percentage = sum/(float)((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1));
	feature_pos[0] = sum/(float)((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1));
	for(int i = 0; i<rlm_matrix.rows; i++){
		float temp_sum = 0;
		for(int j = 0; j<rlm_matrix.cols; j++){
			register float rlm_val = rlm_matrix.at<float>(i,j);
			feature_pos[1] += rlm_val/(float)pow(j+1, 2);
			feature_pos[2] += rlm_val*(float)pow(j+1, 2);
			// features.short_run_emphasis += rlm_val/(float)pow(j+1, 2);
			// features.long_run_emphasis += rlm_val*(float)pow(j+1, 2);
			temp_sum += rlm_val;
			temp_run_sum[j] += rlm_val;
		}
		feature_pos[3] += (float)pow(temp_sum, 2);
		// features.gray_level_non_unformity += (float)pow(temp_sum, 2);
	}
	for(int i = 0; i < rlm_matrix.cols; i++){
		feature_pos[4] += (float)pow(temp_run_sum[i], 2);
	}
	feature_pos[1] /= sum;
	feature_pos[2] /= sum;
	feature_pos[3] /= sum;
	feature_pos[4] /= sum;
	// features.short_run_emphasis /= sum;
	// features.long_run_emphasis /= sum;
	// features.run_length_non_unformity /= sum;
	// features.gray_level_non_unformity /= sum;
	// return features;
}

cv::Mat RLM_3d::getFeatures(){
	// cv::Mat *tmp;
	// tmp = pixel_values;
	// pixel_values = NULL;
	return pixel_values;
}

cv::Mat RLM_3d::addBoarder(cv::Mat *orImage, int *dims){
	int d[3] = {dims[0] + 2*x_radius, dims[1] + 2*y_radius, dims[2] + 2*z_radius};
	cv::Mat tempImage = cv::Mat(3, d, CV_8UC1, cv::Scalar(0));
	int indexes[3];
	for(int i = 0; i < d[0]; i++){
		if(i < x_radius) indexes[0] = x_radius - i;
		else if( i >= dims[0] + x_radius) indexes[0] = 2*(dims[0] - 1) + x_radius - i;
		else indexes[0] = i - x_radius;
		for(int j = 0; j < d[1]; j++){
			if(j < y_radius) indexes[1] = y_radius - j;
			else if( j >= dims[1] + y_radius) indexes[1] = 2*(dims[1] - 1) + y_radius - j;
			else indexes[1] = j - y_radius;
			for(int k = 0; k < d[2]; k++){
				if(k < z_radius) indexes[2] = z_radius - k;
				else if( k >= dims[2] + z_radius) indexes[2] = 2*(dims[2] - 1) + z_radius - k;
				else indexes[2] = k - z_radius;
				tempImage.at<uchar>(i, j, k) = orImage->at<uchar>(indexes[0], indexes[1], indexes[2]);
			}
		}
	}
	return tempImage;
}

void RLM_3d::chunks(int l){
	int n1 = l/num_threads + 1;
	int n2 = l/num_threads;
	int i = 0;
	for(int j = 0; j<l%num_threads; j++){
		t_limits[j]->st = i;
		t_limits[j]->end = i+n1;
		i += n1;
	}
	for(int j = l%num_threads; j<num_threads; j++){
		t_limits[j]->st = i;
		t_limits[j]->end = i+n2;
		i += n2;
	}
}

int RLM_3d::minimum(std::vector<int> lenghts){
	int t = std::numeric_limits<int>::max();
	for(int i = 0; i < lenghts.size(); i++){
		if(lenghts[i] == 0) continue;
		if(lenghts[i] < t) t = lenghts[i];
	}
	return t;
}

RLM_3d::~RLM_3d(){
	// if (pixel_values != NULL)
	// 	delete pixel_values;
	for(int t = 0; t<num_threads; t++){
		free(t_limits[t]);
	}
	free(t_limits);
}