#pragma once



class RLM_3d
{
public:
	RLM_3d(int x_rad, int y_rad, int z_rad, int x_step, int y_step, int z_step, int val_range, int n_threads);
	void computeFeatures(cv::Mat *src, int dims[3], int *mask);
	cv::Mat getFeatures();
	~RLM_3d();

private:
	struct limit_str{
		int st, end;
	};
	void pixel_RLM(unsigned char ***src, int pixel[3], cv::Mat &dst, int direction);
	void chunks(int l);
	void thread_computeFeatures(int thread_id, int dim[3], unsigned char ***src);
	cv::Mat addBoarder(cv::Mat *orImage, int *dims);
	void downsize_matrix(cv::Mat src, unsigned char ***dst, int dim[3]);
	int minimum(std::vector<int> lengths);
	void calculateFeatures(cv::Mat rlm_matrix, float *feature_pos);
	int x_radius, y_radius, z_radius, step_x, step_y, step_z, value_range, num_threads, *map;
	cv::Mat pixel_values;
	RLM_3d::limit_str **t_limits;
};