#pragma once



class RANK
{
public:
	RANK(int n_threads, int d);
	struct rankedFeature{
		int index;
		long double value;
	};
	void getRankedFeatures(float *data, size_t dstep, int _num_points, std::vector<RANK::rankedFeature> &rankedList);
	~RANK();

private:
	void sorted_insert(std::vector<RANK::rankedFeature> &vec, RANK::rankedFeature s);
	void thread_calculate_H(int thread_id, float *data, size_t dstep, long double *H);
	void calculate_maxDiffs(float *data, size_t dstep);
	void calculate_alpha(float *data, size_t dstep);
	void thread_calculate_maxDiffs(int thread_id, float *data, size_t dstep, float *max, float *min);
	long double calculate_H(float *data, size_t dstep);
	long double calculate_S(float *point1, float *point2, int *steps);
	void thread_calculate_alpha(int thread_id, float *data, size_t dstep, float *distances, int *counts);
	int	num_threads, dims, num_points, f, *featsteps, *sFeat;
	float *maxDiffs;
	double alpha;
};