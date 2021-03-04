#pragma once

class IFilter
{
public:
	IFilter(int n_threads, float beta, int d, float r, int s);
	void select_features(cv::Mat &_data);
	void transorm_data(cv::Mat &_data);
	float *getWeights();
	~IFilter();
private:
	void calculate_weights(std::vector<SRANK::rankedFeature> vec);
	void calculate_D(float *data, size_t d_step);
	void part_max_dist(int thread_id, float *data, size_t d_step, float *max_distance);
	float calculate_fei(float *data, size_t d_step);
	void thread_calculate_fei(int thread_id, float *data, size_t d_step, double *fei);
	float threshold, vita, D, *weights, sampling_ratio;
	int num_threads, dims, num_points, num_samples;
};