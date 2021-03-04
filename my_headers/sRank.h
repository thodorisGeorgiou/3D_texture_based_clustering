#pragma once



class SRANK
{
public:
	SRANK(int n_threads, int d, float r, int s);
	struct rankedFeature{
		int index;
		long double value;
	};
	void getRankedFeatures(cv::Mat &data, std::vector<SRANK::rankedFeature> &rankedList);
	~SRANK();

private:
	void sorted_insert(std::vector<SRANK::rankedFeature> &vec, SRANK::rankedFeature s);
	cv::Mat random_samples(cv::Mat &data);
	int	num_threads, dims, num_points, num_samples;
	float ratio;
	long double *OR;
};