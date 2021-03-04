#pragma once

class MITRA_VAR_1
{
public:
	MITRA_VAR_1(int n_threads, float threshold);
	int select_features(cv::Mat &_data);
	~MITRA_VAR_1();
	std::vector<int> kept_features;

private:
	struct dist_struct{
		int index;
		float distance;
	};
	void distance_calculator(cv::Mat &_src, std::vector<std::vector<MITRA_VAR_1::dist_struct> > &dst);
	void thread_cosine_calculator(int thread_id, float *src, size_t step_src, void *dst, int num_features);
	void thread_mici_calculator(int thread_id, float *src, size_t step_src, void *_dst, int num_features);
	void sorted_insert(dist_struct s, std::vector<MITRA_VAR_1::dist_struct> &vec);
	int find_mini(std::vector<std::vector<MITRA_VAR_1::dist_struct> > sorted_distances, bool *checked, bool *deleted);
	int num_threads, dims;
	float threshold;
};