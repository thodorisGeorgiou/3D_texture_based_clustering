#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <thread>
#include "my_kmeans.h"
// This class is copied from matrix.cpp in core module.
class CV_EXPORTS KMeansPPDistanceComputer : public cv::ParallelLoopBody {
public:
KMeansPPDistanceComputer( float *_tdist2, const float *_data, const float *_dist, int _dims,
                        size_t _step, size_t _stepci )
  : tdist2(_tdist2), data(_data), dist(_dist), dims(_dims), step(_step), stepci(_stepci){};

	void operator()( const cv::Range& range ) const
	{
		const int begin = range.start;
		const int end = range.end;

		for ( int i = begin; i<end; i++ )
		{
			tdist2[i] = std::min(cv::normL2Sqr_(data + step*i, data + stepci, dims), dist[i]);
		}
	}

private:
    KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const float *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
};

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const cv::Mat& _data, cv::Mat& _out_centers, int K, cv::RNG& rng, int trials)
{
	int i, j, k, dims = _data.cols, N = _data.rows;
	const float* data = (float*)_data.data;
	size_t step = _data.step/sizeof(data[0]);
	std::vector<int> _centers(K);
	int* centers = &_centers[0];
	std::vector<float> _dist(N*3);
	float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
	double sum0 = 0;
	centers[0] = (unsigned)rng % N;

	for( i = 0; i < N; i++ )
	{
		dist[i] = cv::normL2Sqr_(data + step*i, data + step*centers[0], dims);
		sum0 += dist[i];
	}
	for( k = 1; k < K; k++ )
	{
		double bestSum = DBL_MAX;
		int bestCenter = -1;

		for( j = 0; j < trials; j++ )
		{
			double p = (double)rng*sum0, s = 0;
			for( i = 0; i < N-1; i++ )
				if( (p -= dist[i]) <= 0 )
					break;
			int ci = i;

			cv::parallel_for_(cv::Range(0, N), KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
			for( i = 0; i < N; i++ )
			{
				s += tdist2[i];
			}

			if( s < bestSum )
			{
				bestSum = s;
				bestCenter = ci;
				std::swap(tdist, tdist2);
			}
		}
		centers[k] = bestCenter;
		sum0 = bestSum;
		std::swap(dist, tdist);
	}

	for( k = 0; k < K; k++ )
	{
		const float* src = data + step*centers[k];
		float* dst = _out_centers.ptr<float>(k);
		for( j = 0; j < dims; j++ ){
			dst[j] = src[j];
		}
	}
}

ParrallelKMeans::ParrallelKMeans(int num_thr, int max_itt, float thres, int dimensions, int att){
	num_threads = num_thr;
	max_itters = max_itt;
	threshold = thres;
	dims = dimensions;
	init_trials = 3;
	num_attemps = att;
	limits = (ParrallelKMeans::limit_str **)malloc(num_threads*sizeof(ParrallelKMeans::limit_str *));
	par_Vals = (ParrallelKMeans::parVals **)malloc(num_threads*sizeof(ParrallelKMeans::parVals *));
	partial_error = (double*)malloc(num_threads*sizeof(double));
	for(int t = 0; t<num_threads; t++){
		limits[t] = (ParrallelKMeans::limit_str*)malloc(sizeof(ParrallelKMeans::limit_str));
		par_Vals[t] = (ParrallelKMeans::parVals*)malloc(sizeof(ParrallelKMeans::parVals));
	}
}


void ParrallelKMeans::PartUpdateCenters(int thread_id, const float *_data, size_t step, int *labels){
	ParrallelKMeans::parVals *dst = par_Vals[thread_id];
	int begin = limits[thread_id]->st, end = limits[thread_id]->end;
	dst->centers->setTo(cv::Scalar(0));
	for(int i=0; i<K; i++) dst->numPoints[i] = 0;
	float *_dst = (float*)dst->centers->data;
	size_t stepi = dst->centers->step/sizeof(_dst[0]);
	for(int i = begin; i < end; i++){
		for(int j = 0; j<dims; j++){
			_dst[stepi*labels[i]+j] += _data[step*i+j];
		}
		dst->numPoints[labels[i]]++;
	}
}

void ParrallelKMeans::UpdateLabels(int thread_id, const float *_data, size_t step, int *labels, float *centers, size_t stepc){
	partial_error[thread_id] = 0;
	int begin = limits[thread_id]->st, end = limits[thread_id]->end;
	for(int i = begin; i<end; i++){
		double minDist = std::numeric_limits<double>::max();
		for(int c=0; c<K; c++){
			float dist = cv::normL2Sqr_(_data + step*i, centers + stepc*c, dims);
			if(dist<minDist){
				minDist = dist;
				labels[i] = c;
			}
		}
		partial_error[thread_id] += minDist;
	}	
}

bool ParrallelKMeans::UpdateCenters(cv::Mat &old_centers, int *labels, const float *_data, size_t step){
	int sizes[2] = {K, dims};
	int numPoints[K];
	cv::Mat new_centers(2, sizes, CV_32F, cv::Scalar(0));
	for(int k = 0; k<K; k++){
		numPoints[k] = 0;
		float * center = new_centers.ptr<float>(k);
		for(int t = 0; t < num_threads; t++){
			float *p_sum = par_Vals[t]->centers->ptr<float>(k);
			for(int j = 0; j<dims; j++) center[j] += p_sum[j];
			numPoints[k] += par_Vals[t]->numPoints[k];
		}
	}
	for(int k = 0; k<K; k++){
		if(numPoints[k] == 0){
			int max_k = 0;
			for(int k1 = 1; k1 < K; k1++)
				if (numPoints[max_k] < numPoints[k1]) max_k = k1;
			float maxDist = 0;
			int farthest_point = -1;
			for(int i = 0; i < limits[num_threads - 1]->end; i++){
				if(labels[i] != max_k) continue;
				float *center = old_centers.ptr<float>(max_k);
				float dist = cv::normL2Sqr_(_data + step*i, center, dims);
				if(dist>maxDist){
					maxDist = dist;
					farthest_point = i;
				}
			}
			if(farthest_point!=-1){
				const float *point = _data + step*farthest_point;
				float *center0 = new_centers.ptr<float>(k);
				float *center1 = new_centers.ptr<float>(max_k);
				for(int j = 0; j< dims; j++){
					center0[j] = point[j];
					center1[j] -= point[j];
				}
				numPoints[k]++;
				numPoints[max_k]--;
			}
		}
	}
	bool cought_thres = true;
	for(int k = 0; k<K; k++){
		float *new_center = new_centers.ptr<float>(k);
		float *old_center = old_centers.ptr<float>(k);
		float dist = 0;
		for(int j = 0; j< dims; j++){
			new_center[j] = new_center[j]/numPoints[k];
			if(cought_thres) dist += pow(new_center[j]-old_center[j], 2);
		}
		if(cought_thres){
			dist = sqrtf(dist);
			if(dist>threshold) cought_thres = false;
		}
	}
	cv::swap(old_centers, new_centers);
	return cought_thres;
}

double ParrallelKMeans::cluster(const cv::Mat &data, int k, cv::Mat &centers, cv::Mat &labels){
	K = k;
	cv::Mat new_labels = labels.clone();
	const float *_data = (float*)data.data;
	size_t step = data.step/sizeof(_data[0]);
	cv::RNG& rng = cv::theRNG();
	cv::Mat new_centers(K, dims, CV_32F), best_centers(K, dims, CV_32F);
	std::vector<std::thread> threads;
	chunks(data.rows);
	double error, best_error = std::numeric_limits<float>::max();
	for(int t = 0; t < num_threads; t++){
		int sizes[2] = {K, dims};
		par_Vals[t]->centers = new cv::Mat(2, sizes, CV_32F);
		par_Vals[t]->numPoints = (int*)malloc(K*sizeof(int));		
	}
	for(int attemp = 0; attemp < num_attemps; attemp++){
		std::cout << "\rAttempt: " << attemp << std::flush;
		int * _labels = (int*)new_labels.data;
		for(int itter = 0; itter<max_itters; itter++){
			bool stop = false;
			if(itter == 0){
				generateCentersPP(data, new_centers, K, rng, init_trials);
			}
			else{
				for(int t = 0; t<num_threads; t++){
					threads.push_back(std::thread(&ParrallelKMeans::PartUpdateCenters, this, t, _data, step, _labels));
				}
				for (auto& t: threads) t.join();
				threads.clear();
				stop = UpdateCenters(new_centers, _labels, _data, step);
			}
			float *_centers = (float*)new_centers.data;
			size_t stepc = new_centers.step/sizeof(_centers[0]);
			for(int t = 0; t<num_threads; t++){
				threads.push_back(std::thread(&ParrallelKMeans::UpdateLabels, this, t, _data, step, _labels,
					_centers, stepc));
			}
			for (auto& t: threads) t.join();
			threads.clear();
			if(stop) break;
		}
		error = 0;
		for(int t = 0; t<num_threads; t++) error += partial_error[t];
		if(error<best_error){
			cv::swap(best_centers, new_centers);
			cv::swap(labels, new_labels);
			best_error = error;
		}
	}
	cv::swap(best_centers, centers);
	for(int t = 0; t< num_threads; t++){
		delete par_Vals[t]->centers;
		free(par_Vals[t]->numPoints);
	}
	return best_error;
}

void ParrallelKMeans::chunks(int l){
	int n1 = l/num_threads + 1;
	int n2 = l/num_threads;
	int i = 0;
	for(int j = 0; j<l%num_threads; j++){
		limits[j]->st = i;
		limits[j]->end = i+n1;
		i += n1;
	}
	for(int j = l%num_threads; j<num_threads; j++){
		limits[j]->st = i;
		limits[j]->end = i+n2;
		i += n2;
	}
}

double ParrallelKMeans::calculateClusteringBalance(double lambda, cv::Mat centers, cv::Mat data){
	cv::Mat new_centers(K, dims, CV_32F);
	const float *_data = (float*)data.data;
	size_t step = data.step/sizeof(_data[0]);
	std::vector<std::thread> threads;
	double gamma = 0;
	cv::Mat labels(data.rows, 1, CV_32S, cv::Scalar(0));
	int *_labels = (int*)labels.data;
	for(int t = 0; t<num_threads; t++){
		int sizes[2] = {K, dims};
		par_Vals[t]->centers = new cv::Mat(2, sizes, CV_32F);
		par_Vals[t]->numPoints = (int*)malloc(K*sizeof(int));		
		threads.push_back(std::thread(&ParrallelKMeans::PartUpdateCenters, this, t, _data, step, _labels));
	}
	for (auto& t: threads) t.join();
	threads.clear();
	int sizes[2] = {1, dims};
	cv::Mat g_center(2, sizes, CV_32F, cv::Scalar(0));
	float *_gCenter = (float*)g_center.data;
	for(int t = 0; t < num_threads; t++){
		float *_cntrs = (float*)par_Vals[t]->centers->data;
		for(int j = 0; j<dims; j++){
			_gCenter[j] += _cntrs[j];
		}
	}
	for(int j = 0; j<dims; j++){
		_gCenter[j] = _gCenter[j]/data.rows;
	}
	float *_centers = (float *)centers.data;
	size_t cstep = centers.step/sizeof(_centers[0]);
	for(int c = 0; c<K; c++){
		gamma += cv::normL2Sqr_(_gCenter, _centers + c*cstep, dims);
	}
	double balance = gamma + lambda;
	for(int t = 0; t< num_threads; t++){
		delete par_Vals[t]->centers;
		free(par_Vals[t]->numPoints);
	}
	return balance;
}

ParrallelKMeans::~ParrallelKMeans(){
	for(int t = 0; t<num_threads; t++){
		free(limits[t]);
		free(par_Vals[t]);
	}
	free(limits);
	free(par_Vals);
	free(partial_error);
}