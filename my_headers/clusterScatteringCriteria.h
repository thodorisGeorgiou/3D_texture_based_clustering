#pragma once

class CSCritteria
{
public:
	CSCritteria(int n_threads);
	float calculateCSC(cv::Mat &_centers, cv::Mat &_data, cv::Mat &_labels);
	~CSCritteria();

private:
	void multiplyNadd_vectors(float *vec_1, float *vec_2, cv::Mat &_dst);
	void calculate_p_w(int thread_id, float *data, size_t d_step, float *centers, size_t c_step, int *labels);
	int num_threads, total_num_points, dims, num_centers;
	std::vector<cv::Mat> part_p_w;
	cv::Mat p_w, p_b;
};