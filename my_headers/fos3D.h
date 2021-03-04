#pragma once

class FOS
{
public:
	FOS(int x_rad, int y_rad, int z_rad, int s_x, int s_y, int s_z, int val_range);
	void computeFeatures(cv::Mat *_src, int dims[3], int *mask);
	cv::Mat getFeatures();
	~FOS();

private:
	cv::Mat addBoarder(cv::Mat *orImage, int *dims);
	void downsize_matrix(cv::Mat src, cv::Mat *dst, int dim[3]);
	void calculateFeatures(cv::Mat *glcm_matrix);
	void pixel_gradient(cv::Mat *src, int dim[3], float ***dst);
	void pixel_features(cv::Mat *src, int pixel[3], int pixel_scalar, float ***avg);
	int x_radius, y_radius, z_radius, step_x, step_y, step_z, value_range, *map;
	cv::Mat pixel_values;
};