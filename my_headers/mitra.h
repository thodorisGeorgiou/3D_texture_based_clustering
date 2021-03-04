#pragma once

class MITRA
{
public:
	MITRA(int n_threads, int k);
	int select_features(cv::Mat &_data);
	~MITRA();

private:
	struct dist_struct{
		int index;
		float distance;
	};
	void distance_calculator(cv::Mat &_src, std::vector<std::vector<MITRA::dist_struct> > &dst);
	void thread_distance_calculator(int thread_id, float *src, size_t step_src, void *dst, int calc_step, int num_features, int dims);
	void sorted_insert(dist_struct s, std::vector<MITRA::dist_struct> &vec);
	int find_mini(std::vector<std::vector<MITRA::dist_struct> > sorted_distances, bool *checked, bool *deleted);
	int K, num_threads;
	float threshold;
};