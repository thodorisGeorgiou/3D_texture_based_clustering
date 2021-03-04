class ParrallelKMeans
{
public:
	ParrallelKMeans(int num_thr, int max_itt, float thres, int dimensions, int att);
	double calculateClusteringBalance(double lambda, cv::Mat _centers, cv::Mat _data);
	double cluster(const cv::Mat &data, int k, cv::Mat &centers, cv::Mat &labels);
	~ParrallelKMeans();
private:
	struct limit_str{
		int st, end;
	};
	struct parVals{
		cv::Mat *centers;
		int *numPoints;
	};
	int num_threads, max_itters, dims, init_trials, num_attemps, K;
	double *partial_error;
	float threshold;
	void PartUpdateCenters(int thread_id, const float *_data, size_t step, int *labels);
	void UpdateLabels(int thread_id, const float *_data, size_t step, int *labels, 
		float *centers, size_t stepc);
	bool UpdateCenters(cv::Mat &old_centers, int *labels, const float *_data, size_t step);
	void chunks(int l);
	// cv::Mat centers;
	ParrallelKMeans::limit_str **limits;
	ParrallelKMeans::parVals **par_Vals;
};