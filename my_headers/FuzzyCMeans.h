class FuzzyCMeans
{
public:
	FuzzyCMeans(int num_thr, float thres, int c, int dimensions, int nPoints, int fuzzyness);
	void cluster(const cv::Mat &data, cv::Mat &centers);
	cv::Mat getLabels();
	~FuzzyCMeans();
private:
	struct parVals{
		float **centers;
		float *memberSum;
	};
	int num_threads, dims, C, num_points, m;
	float threshold, **membership;
	// float euclidianSqr(float *point1, float *point2, int dims);
	void PartUpdateCenters(int thread_id, const float *data, size_t step);
	void UpdateMembership(int thread_id, const float *_data, size_t step, float *centers, size_t stepc);
	void calcGlobalCenters(int thread_id, float *centers, size_t stepc);
	void UpdateCenters(float *centers, size_t stepc, const float *data, size_t step);
	bool checkCriterion(float *_centers, size_t stepc, float *_newCenters, size_t stepnc);
	FuzzyCMeans::parVals *par_Vals;
};








