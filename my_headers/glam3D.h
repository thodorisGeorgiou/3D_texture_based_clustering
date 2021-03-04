#pragma once

class GLAM_3d
{
public:
	GLAM_3d(int x_rad, int y_rad, int z_rad, int x_box, int y_box, int z_box, int val_range, int n_threads, int f_calc);
	void computeFeatures(cv::Mat *_src, int dims[3], int *mask);
	cv::Mat getFeatures();
	~GLAM_3d();

private:
	struct limit_str{
		int st, end;
	};
	// struct features_st{
	// 	float contrast_val, dissimilarity_val,  homogeneity_val, second_moment,
	// 	entropy, correlation, clusterShade, clusterProm, maxProb;
	// };
	cv::Mat addBoarder(cv::Mat *orImage, int *dims);
	void pixel_GLAM(unsigned char ***src, int pixel[3], cv::Mat &dst);
	void normalize_matrix(cv::Mat &src, cv::Mat &dst);
	void chunks(int l);
	void thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src);
	void downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]);
	void calculateFeatures_subset(cv::Mat glam_matrix, float *feature_pos);
	void calculateFeatures_all(cv::Mat glam_matrix, float *feature_pos);
	int x_radius, y_radius, z_radius, b_size[3], value_range, num_threads, feat_calculator, num_feats[2] = {8, 15}, *map;
	// cv::Mat **t_pixel_values;
	cv::Mat pixel_values;
	GLAM_3d::limit_str **t_limits;
};