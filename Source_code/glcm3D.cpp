#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <math.h>
#include "glcm3D.h"

GLCM_3d::GLCM_3d(int x_rad, int y_rad, int z_rad, int x_step, int y_step, int z_step, int val_range, int n_threads, int f_calc){
	x_radius = x_rad;
	y_radius = y_rad;
	z_radius = z_rad;
	step_x = x_step;
	step_y = y_step;
	step_z = z_step;
	value_range = val_range;
	num_threads = n_threads;
	feat_calculator = f_calc;
	t_limits = (GLCM_3d::limit_str **)malloc(num_threads*sizeof(GLCM_3d::limit_str *));
	for(int t = 0; t<num_threads; t++){
		t_limits[t] = (GLCM_3d::limit_str*)malloc(sizeof(GLCM_3d::limit_str));
	}
}

void GLCM_3d::pixel_GLCM(unsigned char ***src, int pixel[3], cv::Mat &dst, int direction){
	int sizes[] = {value_range, value_range};
	int limits_i[] = {x_radius, x_radius - step_x, x_radius, x_radius, x_radius, x_radius, 
						x_radius, x_radius - step_x, x_radius - step_x, x_radius, x_radius, x_radius - step_x,
						x_radius, x_radius - step_x, x_radius, x_radius - step_x, x_radius, x_radius,
						x_radius, x_radius - step_x, x_radius - step_x, x_radius, x_radius, x_radius};
	int limits_j[] = {y_radius, y_radius, y_radius, y_radius - step_y, y_radius, y_radius, 
						y_radius, y_radius - step_y, y_radius, y_radius - step_y, y_radius - step_y, y_radius,
						y_radius, y_radius - step_y, y_radius, y_radius, y_radius, y_radius - step_y,
						y_radius - step_y, y_radius, y_radius, y_radius, y_radius, y_radius - step_y};
	int limits_k[] = {z_radius, z_radius, z_radius, z_radius, z_radius, z_radius - step_z, 
						z_radius, z_radius - step_z, z_radius, z_radius - step_z, z_radius, z_radius - step_z, 
						z_radius, z_radius, z_radius, z_radius - step_z, z_radius, z_radius - step_z,
						z_radius, z_radius, z_radius, z_radius - step_z, z_radius - step_z, z_radius};
	int neighbor_i[] = {step_x, 0, 0, step_x, -step_x, step_x, step_x, step_x, 0, step_x, -step_x, 0};
	int neighbor_j[] = {0, step_y, 0, step_y, step_y, -step_y, step_y, 0, step_y, -step_y, 0, step_y};
	int neighbor_k[] = {0, 0, step_z, step_z, step_z, step_z, 0, step_z, step_z, 0, step_z, -step_z};
	int x, y;
	dst = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	for(int k=pixel[2]-limits_k[2*direction]; k<=pixel[2]+limits_k[2*direction+1]; k++){
		for(int i=pixel[0]-limits_i[2*direction]; i<=pixel[0]+limits_i[2*direction+1]; i++){
			for(int j=pixel[1]-limits_j[2*direction]; j<=pixel[1]+limits_j[2*direction+1]; j++){
				int c_pix[3];
				c_pix[0] = i + neighbor_i[direction];
				c_pix[1] = j + neighbor_j[direction];
				c_pix[2] = k + neighbor_k[direction];
				x = (int)src[i][j][k];
				y = (int)src[c_pix[0]][c_pix[1]][c_pix[2]];
				dst.at<float>(x, y) += 1;
				dst.at<float>(y, x) += 1;
			}
		}
	}
	normalize_matrix(dst, dst);
}

void GLCM_3d::normalize_matrix(cv::Mat &src, cv::Mat &dst){
	float sum = cv::sum(src)[0];
	for(int i=0; i < src.rows; i++){
		for(int j=0; j < src.cols; j++){
			dst.at<float>(i,j) = src.at<float>(i,j)/sum;
		}
	}
}

void GLCM_3d::downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]){
	int scale = 256/value_range;
	for(int i=0; i < dim[0]; i++){
		for(int j=0; j < dim[1]; j++){
			for(int k=0; k < dim[2]; k++){
				dst[i][j][k] = (unsigned char)(src.at<uchar>(i,j,k)/scale);
			}
		}
	}
}

void GLCM_3d::computeFeatures(cv::Mat *_src, int dims[3], int *mask){
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
	int sizes[] = {maskCount, num_feats[feat_calculator]*12};
	pixel_values = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));

	for(int t = 0; t<num_threads; t++){
		threads.push_back(std::thread(&GLCM_3d::thread_computeFeatures, this, t, dims, tmp_src));
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

void GLCM_3d::thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src){
	int start=t_limits[thread_id]->st;
	int end=t_limits[thread_id]->end;
	float *data = (float*)pixel_values.data;
	size_t d_step = pixel_values.step/sizeof(data[0]);
	for(int p = start; p < end; p++){
		float *feature_pos = data + p*d_step;
		int mod = map[p] % (dim[1]*dim[0]);
		int pixel[] = {mod % dim[0] + x_radius, mod / dim[0] + y_radius, map[p] / (dim[1]*dim[0]) + z_radius};
		for(int direction = 0; direction<12; direction++){
			cv::Mat glcm_matrix;
			pixel_GLCM(src, pixel, glcm_matrix, direction);
			switch(feat_calculator){
				case 0:
					calculateFeatures_subset(glcm_matrix, feature_pos);
					feature_pos += 8;
					break;
				case 1:
					calculateFeatures_all(glcm_matrix, feature_pos);
					feature_pos += 15;
					break;
				default:
					std::cout << "Wrong feature calculator, exiting.." << std::endl;
					exit(0);
			}
		}
	}
}

void GLCM_3d::calculateFeatures_all(cv::Mat glcm_matrix, float *feature_pos){
	register float m = 0, m_x = 0, m_y = 0, s_x = 0, s_y = 0, m_xpy = 0;
	register float p_x[glcm_matrix.rows], p_y[glcm_matrix.rows], p_xky[2*glcm_matrix.rows - 1], p_xpy[glcm_matrix.rows];
	register float hx = 0, hy = 0, hxy1 = 0, hxy2 = 0;
	std::fill_n(p_x, glcm_matrix.rows, m);
	std::fill_n(p_y, glcm_matrix.rows, m);
	std::fill_n(p_xky, 2*glcm_matrix.rows-1, m);
	std::fill_n(p_xpy, glcm_matrix.rows, m);
	for(int i = 1; i<=glcm_matrix.rows; i++){
		for(int j = 1; j<=glcm_matrix.cols; j++){
			register int x = i -1, y = j -1;
			register float glcm_val = glcm_matrix.at<float>(x,y);
			p_x[x] += glcm_val;
			p_y[y] += glcm_val;
			m += (i+j)*glcm_val;
			p_xky[i+j-2] += glcm_val;
			p_xpy[(int)(abs(i-j))] += glcm_val;
			feature_pos[0] += (float)pow(glcm_val, 2);
			feature_pos[1] += (float)(pow(i - j, 2)*glcm_val);
			feature_pos[2] += (float)(i*j*glcm_val);
			feature_pos[4] += (float)(glcm_val/(1+(float)pow(i - j, 2)));
			double lg  = log10(glcm_val);
			if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
			feature_pos[8] -= (float)(glcm_val*lg);
			feature_pos[13] += (float)(abs(i - j)*glcm_val);
			if(glcm_val>feature_pos[14]) feature_pos[14] = glcm_val;
		}
	}
	m = m / 2;
	for(int i = 1; i<=glcm_matrix.rows; i++){
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
		for(int j = 1; j<=glcm_matrix.rows; j++){
			register int y = j - 1;
			register float glcm_val = glcm_matrix.at<float>(x,y);
			double lg  = log10(p_x[x]*p_y[y]);
			if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
			hxy1 -= (float)(glcm_val*lg);
			hxy2 -= (float)(p_x[x]*p_y[y]*lg);
			feature_pos[3] += (float)(pow(i - m, 2)*glcm_val);
		}
	}
	feature_pos[9] -= m_xpy*m_xpy;
	s_x = sqrtf(s_x - (m_x*m_x));
	s_y = sqrtf(s_y - (m_y*m_y));
	feature_pos[2] = (float)(feature_pos[2] - (m_x*m_y))/(float)(s_x*s_y);
	for(int k = 2; k<=2*glcm_matrix.rows; k++){
		register int x = k-2;
		double lg  = log10(p_xky[x]);
		if(lg<std::numeric_limits<float>::min()) lg = std::numeric_limits<float>::min();
		feature_pos[7] -= (float)(p_xky[x]*lg);
		feature_pos[5] += p_xky[x]*k;
	}
	for(int k = 2; k<=2*glcm_matrix.rows; k++)
		feature_pos[6] += (float)(pow(k-feature_pos[7], 2)*p_xky[k-2]);
	if(hx > hy)
		feature_pos[11] = (feature_pos[8] - hxy1)/hx;
	else
		feature_pos[11] = (feature_pos[8] - hxy1)/hy;
	feature_pos[12] = sqrtf(1 - (float)exp((-2)*(hxy2 - feature_pos[8])));
}

void GLCM_3d::calculateFeatures_subset(cv::Mat glcm_matrix, float *feature_pos){
	register float m_x = 0, m_y = 0, s_x = 0, s_y = 0;
	for(int i = 0; i<glcm_matrix.rows; i++){
		for(int j = 0; j<glcm_matrix.cols; j++){
			register float glcm_val = glcm_matrix.at<float>(i,j);
			m_x += glcm_val*i;
			m_y += glcm_val*j;
			// s_x += glcm_val*i*i;
			// s_y += glcm_val*j*j;
			feature_pos[0] += (float)(pow(i - j, 2)*glcm_val);
			feature_pos[1] += (float)(abs(i - j)*glcm_val);
			feature_pos[2] += (float)(glcm_val/(1+(float)pow(i - j, 2)));
			feature_pos[3] += (float)pow(glcm_val, 2);
			double lg  = log10(glcm_val);
			if(lg<-std::numeric_limits<float>::max()) lg = 0;
			feature_pos[4] += (float)(glcm_val*lg);
			if(glcm_val>feature_pos[5]) feature_pos[5] = glcm_val;
		}
	}
	// s_x = sqrtf(s_x - (float)pow(m_x, 2));
	// s_y = sqrtf(s_y - (float)pow(m_y, 2));
	for(int i = 0; i<glcm_matrix.rows; i++){
		for(int j = 0; j<glcm_matrix.cols; j++){
			register float glcm_val = glcm_matrix.at<float>(i,j);
			// features.correlation += (float)((i - m_x)*(j - m_y)*glcm_val/(float)(s_x*s_y));
			feature_pos[6] += (float)(pow(i+j-m_x-m_y, 3)*glcm_val);
			feature_pos[7] += (float)(pow(i+j-m_x-m_y, 4)*glcm_val);
		}
	}
	// return features;
}

cv::Mat GLCM_3d::addBoarder(cv::Mat *orImage, int *dims){
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

cv::Mat GLCM_3d::getFeatures(){
	return pixel_values;
}

void GLCM_3d::chunks(int l){
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

GLCM_3d::~GLCM_3d(){
	// pixel_values.release();
	for(int t = 0; t<num_threads; t++){
		free(t_limits[t]);
	}
	free(t_limits);
}