#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "math.h"
#include "fos3D.h"

FOS::FOS(int x_rad, int y_rad, int z_rad, int s_x, int s_y, int s_z, int val_range){
	x_radius = x_rad - s_x;
	y_radius = y_rad - s_y;
	z_radius = z_rad - s_z;
	step_x = s_x;
	step_y = s_y;
	step_z = s_z;
	value_range = val_range;
}

void FOS::downsize_matrix(cv::Mat src, cv::Mat *dst, int dim[3]){
	int scale = 256/value_range;
	for(int i=0; i < dim[0]; i++){
		for(int j=0; j < dim[1]; j++){
			for(int k=0; k < dim[2]; k++){
				dst->at<uchar>(i,j,k) = src.at<uchar>(i,j,k)/scale;
			}
		}
	}
}

void FOS::pixel_features(cv::Mat *src, int pixel[3], int pixel_scalar, float ***agv){
	register float mgl = 0, vgl = 0, magv = 0, vagv = 0;
	for(int k=pixel[2] - z_radius; k <= pixel[2] + z_radius; k++){
		for(int j=pixel[1] - y_radius;j <= pixel[1] + y_radius; j++){
			for(int i=pixel[0] - x_radius;i <= pixel[0] + x_radius; i++){
				vgl += (float)pow(src->at<uchar>(i, j, k), 2);
				mgl += (float)src->at<uchar>(i, j, k);
				vagv += (float)pow(agv[i - step_x][j - step_y][k - step_z], 2);
				magv += agv[i - step_x][j - step_y][k - step_z];
			}
		}
	}
	mgl = mgl/((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1));
	vgl = vgl/((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1)) - (float)pow(mgl, 2);
	magv = magv/((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1));
	vagv = vagv/((2*x_radius+1)*(2*y_radius+1)*(2*z_radius+1)) - (float)pow(magv, 2);
	pixel_values.at<float>(pixel_scalar, 0) = mgl;
	pixel_values.at<float>(pixel_scalar, 1) = vgl;
	// pixel_values.at<float>(pixel_scalar, 2) = agv[pixel[0] - step_x][pixel[1] - step_y][pixel[2] - step_z];
	pixel_values.at<float>(pixel_scalar, 2) = magv;
	pixel_values.at<float>(pixel_scalar, 3) = vagv;
}

void FOS::pixel_gradient(cv::Mat *src, int dim[3], float ***dst){
	int radii[3] = {step_x, step_y, step_z};
	for(int i = 0; i < dim[0]; i++){
		for(int j = 0; j < dim[1]; j++){
			for(int k = 0; k < dim[2]; k++){
				dst[i][j][k] = sqrtf((float)pow((float)src->at<uchar>(i+ step_x + radii[0], j + radii[1], k + radii[2]) - (float)src->at<uchar>(i- step_x + radii[0], j + radii[1], k + radii[2]), 2) +
				(float)pow((float)src->at<uchar>(i + radii[0], j+step_y + radii[1], k + radii[2]) - (float)src->at<uchar>(i + radii[0], j-step_y + radii[1], k + radii[2]), 2) +
				(float)pow((float)src->at<uchar>(i + radii[0], j + radii[1], k+step_z + radii[2]) - (float)src->at<uchar>(i + radii[0], j + radii[1], k-step_z + radii[2]), 2));
			}
		}
	}
}

void FOS::computeFeatures(cv::Mat *_src, int dims[3], int *mask){
	int dim[3] = {dims[0] + 2*(x_radius + step_x), dims[1] + 2*(y_radius + step_y), dims[2] + 2*(z_radius + step_z)};
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
	int sizes[] = {maskCount, 4};
	pixel_values = cv::Mat(2, sizes, CV_32F, cv::Scalar(0));
	float ***agv;
	int avgDims[3] = {dims[0] + 2*x_radius, dims[1] + 2*y_radius, dims[2] + 2*z_radius};
	agv = (float***)malloc((avgDims[0])*sizeof(float**));
	for(int i = 0; i < avgDims[0]; i++){
		agv[i] = (float**)malloc((avgDims[1])*sizeof(float*));
		for(int j = 0; j < avgDims[1]; j++)
			agv[i][j] = (float*)malloc((avgDims[2])*sizeof(float));
	}
	cv::Mat *tmp_src = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
	downsize_matrix(src, tmp_src, dim);
	pixel_gradient(tmp_src, avgDims, agv);
	for(int p = 0; p < maskCount; p++){
		int mod = map[p] % (dims[1]*dims[0]);
		int pixel[] = {mod % dims[0] + x_radius + step_x, mod / dims[0] + y_radius + step_y, map[p] / (dims[1]*dims[0]) + z_radius + step_z};
		// std::cout << pixel[0] << " " <<pixel[1] << " " <<  pixel[2] << std::endl;
		pixel_features(tmp_src, pixel, p, agv);
	}
	delete tmp_src;
	delete[] map;
}

cv::Mat FOS::addBoarder(cv::Mat *orImage, int *dims){
	int radii[3] = {x_radius + step_x, y_radius + step_y, z_radius + step_z};
	int d[3] = {dims[0] + 2*radii[0], dims[1] + 2*radii[1], dims[2] + 2*radii[2]};
	cv::Mat tempImage = cv::Mat(3, d, CV_8UC1, cv::Scalar(0));
	int indexes[3];
	for(int i = 0; i < d[0]; i++){
		if(i < radii[0]) indexes[0] = radii[0] - i;
		else if( i >= dims[0] + radii[0]) indexes[0] = 2*(dims[0] - 1) + radii[0] - i;
		else indexes[0] = i - radii[0];
		for(int j = 0; j < d[1]; j++){
			if(j < radii[1]) indexes[1] = radii[1] - j;
			else if( j >= dims[1] + radii[1]) indexes[1] = 2*(dims[1] - 1) + radii[1] - j;
			else indexes[1] = j - radii[1];
			for(int k = 0; k < d[2]; k++){
				if(k < radii[2]) indexes[2] = radii[2] - k;
				else if( k >= dims[2] + radii[2]) indexes[2] = 2*(dims[2] - 1) + radii[2] - k;
				else indexes[2] = k - radii[2];
				tempImage.at<uchar>(i, j, k) = orImage->at<uchar>(indexes[0], indexes[1], indexes[2]);
			}
		}
	}
	return tempImage;
}

cv::Mat FOS::getFeatures(){
	return pixel_values;
}

FOS::~FOS(){
}