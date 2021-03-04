#pragma once



class GLCM_3d
{
public:
	GLCM_3d(int x_rad, int y_rad, int z_rad, int x_step, int y_step, int z_step, int val_range, int n_threads, int f_calc);
	void computeFeatures(cv::Mat *src, int dim[3], int *mask);
	cv::Mat getFeatures();
	~GLCM_3d();

private:
	struct limit_str{
		int st, end;
	};
	// struct features_st{
	// 	float contrast_val, dissimilarity_val,  homogeneity_val, second_moment,
	// 	entropy, correlation, clusterShade, clusterProm, maxProb;
	// };
	void pixel_GLCM(unsigned char ***src, int pixel[3], cv::Mat &dst, int direction);
	void normalize_matrix(cv::Mat &src, cv::Mat &dst);
	void chunks(int l);
	void thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src);
	// cv::Mat *getSlice(cv::Mat *src, int dim[3], int k);
	cv::Mat addBoarder(cv::Mat *orImage, int *dims);
	void downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]);
	void calculateFeatures_all(cv::Mat glcm_matrix, float *feature_pos);
	void calculateFeatures_subset(cv::Mat glcm_matrix, float *feature_pos);
	int x_radius, y_radius, z_radius, step_x, step_y, step_z, value_range, num_threads, feat_calculator, num_feats[2] = {8, 15}, *map;
	// std::vector<cv::Mat> t_pixel_values;
	cv::Mat pixel_values;
	GLCM_3d::limit_str **t_limits;
};