#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "glam3D.h"

GLAM_3d::GLAM_3d(int x_rad, int y_rad, int z_rad, int x_box, int y_box, int z_box, int val_range, int n_threads, int f_calc){
	x_radius = x_rad;
	y_radius = y_rad;
	z_radius = z_rad;
	b_size[0] = x_box;
	b_size[1] = y_box;
	b_size[2] = z_box;
	value_range = val_range;
	num_threads = n_threads;
	feat_calculator = f_calc;
	t_limits = (GLAM_3d::limit_str **)malloc(num_threads*sizeof(GLAM_3d::limit_str *));
	// t_pixel_values = (cv::Mat **)malloc(num_threads*sizeof(cv::Mat *));
	for(int t = 0; t<num_threads; t++){
		t_limits[t] = (GLAM_3d::limit_str*)malloc(sizeof(GLAM_3d::limit_str));
	}
	// pixel_values = NULL;
}

void GLAM_3d::pixel_GLAM(unsigned char ***src, int pixel[3], cv::Mat &dst){
	int sizes[] = {value_range, value_range};
	int x, y;
	dst = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	for(int k=pixel[2]-z_radius; k<=pixel[2]+z_radius; k++){
		for(int i=pixel[0]-x_radius; i<=pixel[0]+x_radius; i++){
			for(int j=pixel[1]-y_radius; j<=pixel[1]+y_radius; j++){
				x = (int)src[i][j][k];
				for(int a = i - b_size[0]; a <= i + b_size[0]; a++){
					for(int b = j - b_size[1]; b <= j + b_size[1]; b++){
						for(int c = k - b_size[2]; c <= k + b_size[2]; c++){
							if ((a < pixel[0] - x_radius) || (a > pixel[0] + x_radius) || (b < pixel[1] - y_radius) ||
								(b > pixel[1] + y_radius) || (c < pixel[2] - z_radius) || (c > pixel[2] + z_radius)) continue;
							y = (int)src[a][b][c];
							dst.at<float>(x, y) += 1;
						}
					}
				}
			}
		}
	}
	normalize_matrix(dst, dst);
}

void GLAM_3d::normalize_matrix(cv::Mat &src, cv::Mat &dst){
	float sum = cv::sum(src)[0];
	for(int i=0; i < src.rows; i++){
		for(int j=0; j < src.cols; j++){
			dst.at<float>(i,j) = src.at<float>(i,j)/sum;
		}
	}
}

void GLAM_3d::downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]){
	int scale = 256/value_range;
	for(int i=0; i < dim[0]; i++){
		for(int j=0; j < dim[1]; j++){
			for(int k=0; k < dim[2]; k++){
				dst[i][j][k] = src.at<uchar>(i,j,k)/scale;
			}
		}
	}
}

void GLAM_3d::computeFeatures(cv::Mat *_src, int dims[3], int *mask){
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
	int sizes[] = {maskCount, num_feats[feat_calculator]};
	pixel_values = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	for(int t = 0; t<num_threads; t++){
		threads.push_back(std::thread(&GLAM_3d::thread_computeFeatures, this, t, dims, tmp_src));
	}
	for (auto& t: threads) t.join();
	std::cout << "All threads finished" << std::endl;
	for(int i = 0; i < dim[0]; i++){
		for(int j = 0; j < dim[1]; j++){
			free(tmp_src[i][j]);
		}
		free(tmp_src[i]);
	}
	free(tmp_src);
	delete[] map;
}

void GLAM_3d::thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src){
	int start=t_limits[thread_id]->st;
	int end=t_limits[thread_id]->end;
	float *data = (float*)pixel_values.data;
	size_t d_step = pixel_values.step/sizeof(data[0]);
	for(int p = start; p < end; p++){
		float *feature_pos = data + p*d_step;
		int mod = map[p] % (dim[1]*dim[0]);
		int pixel[] = {mod % dim[0] + x_radius, mod / dim[0] + y_radius, map[p] / (dim[1]*dim[0]) + z_radius};
		cv::Mat glam_matrix;
		pixel_GLAM(src, pixel, glam_matrix);
		switch(feat_calculator){
			case 0:
				calculateFeatures_subset(glam_matrix, feature_pos);
				break;
			case 1:
				calculateFeatures_all(glam_matrix, feature_pos);
				break;
			default:
				std::cout << "Wrong feature calculator, exiting.." << std::endl;
				exit(0);
		}
	}
}

void GLAM_3d::calculateFeatures_all(cv::Mat glam_matrix, float *feature_pos){
	// GLCM_3d::features_st features;
	// features.contrast_val = 0;
	// features.dissimilarity_val = 0;
	// features.homogeneity_val = 0;
	// features.second_moment = 0;
	// features.entropy = 0;
	// // features.correlation = 0;
	// // features.clusterShade = 0;
	// // features.clusterProm = 0;
	// features.maxProb = 0;
	// feature_pos[0] = 0;
	// feature_pos[1] = 0;
	// feature_pos[2] = 0;
	// feature_pos[3] = 0;
	// feature_pos[4] = 0;
	// feature_pos[5] = 0;
	// register float sum = cv::sum(src)[0];
	register float m = 0, m_x = 0, m_y = 0, s_x = 0, s_y = 0, m_xpy = 0;
	register float p_x[glam_matrix.rows], p_y[glam_matrix.rows], p_xky[2*glam_matrix.rows - 1], p_xpy[glam_matrix.rows];
	register float hx = 0, hy = 0, hxy1 = 0, hxy2 = 0;
	std::fill_n(p_x, glam_matrix.rows, m);
	std::fill_n(p_y, glam_matrix.rows, m);
	std::fill_n(p_xky, 2*glam_matrix.rows-1, m);
	std::fill_n(p_xpy, glam_matrix.rows, m);
	for(int i = 1; i<=glam_matrix.rows; i++){
		for(int j = 1; j<=glam_matrix.cols; j++){
			register int x = i -1, y = j -1;
			register float glam_val = glam_matrix.at<float>(x,y);
			p_x[x] += glam_val;
			p_y[y] += glam_val;
			m += (i+j)*glam_val;
			p_xky[i+j-2] += glam_val;
			p_xpy[(int)(abs(i-j))] += glam_val;
			feature_pos[0] += (float)pow(glam_val, 2);
			feature_pos[1] += (float)(pow(i - j, 2)*glam_val);
			feature_pos[2] += (float)(i*j*glam_val);
			feature_pos[4] += (float)(glam_val/(1+(float)pow(i - j, 2)));
			double lg  = log10(glam_val);
			if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
			feature_pos[8] -= (float)(glam_val*lg);
			feature_pos[13] += (float)(abs(i - j)*glam_val);
			if(glam_val>feature_pos[14]) feature_pos[14] = glam_val;
		}
	}
	m = m / 2;
	for(int i = 1; i<=glam_matrix.rows; i++){
		register int x = i -1;
		double lg  = log10(p_x[x]);
		if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
		hx -= (float)(p_x[x]*lg);
		lg  = log10(p_y[x]);
		if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
		hy -= (float)(p_y[x]*lg);
		lg  = log10(p_xpy[x]);
		if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
		feature_pos[10] -= (float)(p_xpy[x]*lg);
		m_x += i*p_x[x];
		m_y += i*p_y[x];
		s_x += (float)(pow(i, 2)*p_x[x]);
		s_y += (float)(pow(i, 2)*p_y[x]);
		m_xpy += p_xpy[x]*i;
		feature_pos[9] += p_xpy[x]*i*i;
		for(int j = 1; j<=glam_matrix.rows; j++){
			register int y = j - 1;
			register float glam_val = glam_matrix.at<float>(x,y);
			double lg  = log10(p_x[x]*p_y[y]);
			if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
			hxy1 -= (float)(glam_val*lg);
			hxy2 -= (float)(p_x[x]*p_y[y]*lg);
			feature_pos[3] += (float)(pow(i - m, 2)*glam_val);
		}
	}
	feature_pos[9] -= m_xpy*m_xpy;
	s_x = sqrtf(s_x - (m_x*m_x));
	s_y = sqrtf(s_y - (m_y*m_y));
	feature_pos[2] = (float)(feature_pos[2] - (m_x*m_y))/(float)(s_x*s_y);
	for(int k = 2; k<=2*glam_matrix.rows; k++){
		register int x = k-2;
		double lg  = log10(p_xky[x]);
		if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
		feature_pos[7] -= (float)(p_xky[x]*lg);
		feature_pos[5] += p_xky[x]*k;
	}
	for(int k = 2; k<=2*glam_matrix.rows; k++)
		feature_pos[6] += (float)(pow(k-feature_pos[7], 2)*p_xky[k-2]);
	if(hx > hy)
		feature_pos[11] = (feature_pos[8] - hxy1)/hx;
	else
		feature_pos[11] = (feature_pos[8] - hxy1)/hy;
	feature_pos[12] = sqrtf(1 - (float)exp((-2)*(hxy2 - feature_pos[8])));
}

void GLAM_3d::calculateFeatures_subset(cv::Mat glam_matrix, float *feature_pos){
	// GLAM_3d::features_st features;
	// features.contrast_val = 0;
	// features.dissimilarity_val = 0;
	// features.homogeneity_val = 0;
	// features.second_moment = 0;
	// features.entropy = 0;
	// features.correlation = 0;
	// features.clusterShade = 0;
	// features.clusterProm = 0;
	// features.maxProb = 0;
	float m_x = 0, m_y = 0/*, s_x = 0, s_y = 0*/;
	for(int i = 0; i<glam_matrix.rows; i++){
		for(int j = 0; j<glam_matrix.cols; j++){
			register float glam_val = glam_matrix.at<float>(i,j);
			m_x += glam_val*i;
			m_y += glam_val*j;
			// s_x += glam_val*i*i;
			// s_y += glam_val*j*j;
			feature_pos[0] += (float)pow(i - j, 2)*glam_val;
			feature_pos[1] += abs(i - j)*glam_val;
			feature_pos[2] += glam_val/(float)(1+(float)pow(i - j, 2));
			feature_pos[3] += (float)pow(glam_val, 2);
			double lg  = log10(glam_val);
			if(lg<-std::numeric_limits<float>::max()) lg = 0;
			feature_pos[4] += (float)(glam_val*lg);
			if(glam_val>feature_pos[5]) feature_pos[5] = glam_val;
			// features.contrast_val += (float)pow(i - j, 2)*glam_val;
			// features.dissimilarity_val += abs(i - j)*glam_val;
			// features.homogeneity_val += glam_val/(float)(1+(float)pow(i - j, 2));
			// features.second_moment += (float)pow(glam_val, 2);
			// double lg  = log10(glam_val);
			// if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
			// features.entropy += (float)(glam_val*lg);
			// if(glam_val>features.maxProb) features.maxProb = glam_val;
		}
	}
	// s_x = sqrtf(s_x - (float)pow(m_x, 2));
	// s_y = sqrtf(s_y - (float)pow(m_y, 2));
	for(int i = 0; i<glam_matrix.rows; i++){
		for(int j = 0; j<glam_matrix.cols; j++){
			register float glam_val = glam_matrix.at<float>(i,j);
			// features.correlation += (float)((i - m_x)*(j - m_y)*glam_val/(float)(s_x*s_y));
			feature_pos[6] += (float)pow(i+j-m_x-m_y, 3)*glam_val;
			feature_pos[7] += (float)pow(i+j-m_x-m_y, 4)*glam_val;
			// features.clusterShade += (float)pow(i+j-m_x-m_y, 3)*glam_val;
			// features.clusterProm += (float)pow(i+j-m_x-m_y, 4)*glam_val;
		}
	}
	// return features;
}

cv::Mat GLAM_3d::addBoarder(cv::Mat *orImage, int *dims){
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

cv::Mat GLAM_3d::getFeatures(){
	// cv::Mat *tmp;
	// tmp = pixel_values;
	// pixel_values = NULL;
	return pixel_values;
}

void GLAM_3d::chunks(int l){
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

GLAM_3d::~GLAM_3d(){
	// if (pixel_values != NULL)
	// 	delete pixel_values;
	for(int t = 0; t<num_threads; t++){
		free(t_limits[t]);
	}
	free(t_limits);
}