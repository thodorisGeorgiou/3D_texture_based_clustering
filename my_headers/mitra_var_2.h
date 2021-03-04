#pragma once

class MITRA_VAR_2
{
public:
	MITRA_VAR_2(int n_threads, float threshold);
	int select_features(cv::Mat &_data, cv::Mat &samples);
	~MITRA_VAR_2();
	std::vector<int> kept_features;

private:
	struct dist_struct{
		int index;
		float distance;
	};
	void create_distance_histogramm(float *data, size_t data_step, int dims, int feature, long *histogramm, int num_threads);
	void find_max_distance(int thread_id, int num_threads, float *data, size_t data_step, int dims, int feature, float *max_distance, float *min_distance);
	void thread_compute_histogramm(int thread_id, int num_threads, float *data, size_t data_step, int dims, int feature, long *histogramm, float max_distance);
	void checknDiscard(std::vector<MITRA_VAR_2::dist_struct> &entropy, int feature, cv::Mat &_data);
	void thread_distance_calculator(int thread_id, float *src, size_t step_src, float *_dst, int dims, int calc_step, int cFeature, void *_entropy);
	float feature_entropy(cv::Mat _data, int feature, int num_threads);	
	int serial_sum(int st, int end);
	void sorted_insert(dist_struct s, std::vector<MITRA_VAR_2::dist_struct> &vec);
	int num_threads;
	float threshold;
};