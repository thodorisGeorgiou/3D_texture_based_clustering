#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <cstring>
#include <cmath>
#include <rank.h>

RANK::RANK(int n_threads, int d){
	num_threads = n_threads;
	dims = d;
	featsteps = new int[dims];
	sFeat = new int[dims];
	std::fill_n(featsteps, dims, 1);
	std::fill_n(sFeat, dims, 0);
	sFeat[0] = 1;
}

void RANK::getRankedFeatures(float *data, size_t dstep, int _num_points, std::vector<RANK::rankedFeature> &rankedList){
	num_points = _num_points;
	calculate_maxDiffs(data, dstep);
	for(f = 0; f<dims; f++){
		if(f != 0)
			featsteps[f - 1] = 2;
		RANK::rankedFeature rankS;
		calculate_alpha(data, dstep);
		rankS.value = calculate_H(data, dstep);
		if(f != 0)
			featsteps[f - 1] = 1;
		rankS.index = f;
		sorted_insert(rankedList, rankS);
	}
	delete[] maxDiffs;
}

void RANK::sorted_insert(std::vector<RANK::rankedFeature> &vec, RANK::rankedFeature s){
	int temp = 0;
	register float value = s.value;
	while(temp < vec.size()){
		if(value > vec[temp].value)
			break;
		temp++;
	}
	vec.insert(vec.begin()+temp, s);
}

long double RANK::calculate_H(float *data, size_t dstep){
	std::vector<std::thread> threads;
	long double partial_H[num_threads];
	for(int t = 0; t < num_threads; t++)
		threads.push_back(std::thread(&RANK::thread_calculate_H, this, t, data, dstep, partial_H+t));
	for(auto& t: threads) t.join();
	long double H = 0;
	for(int t = 0; t < num_threads; t++) H += partial_H[t];
	return H;
}

void RANK::thread_calculate_H(int thread_id, float *data, size_t dstep, long double *H){
	int steps[dims];
	std::memcpy(steps, featsteps, dims*sizeof(int));
	*H = 0;
	for(int point1 = thread_id; point1 < num_points - 1; point1 += num_threads){
		for(int point2 = point1+1; point2 < num_points; point2++){
			long double S = calculate_S(data + point1 * dstep, data + point2 * dstep, steps);
			(*H) += (S*exp(S) + (1.0 - S)*exp(1.0 - S));
		}
	}
}


long double RANK::calculate_S(float *point1, float *point2, int *steps){
	long double D = 0;
	for(int d = sFeat[f]; d<dims; d += steps[d]){
		if(maxDiffs[d] != 0){
			long double kapa = (long double)(point1[d] - point2[d])/maxDiffs[d];
			D += kapa*kapa;
		}
		else
			D += 1;
	}
	return exp(-alpha*sqrtl(D));
}

void RANK::calculate_maxDiffs(float *data, size_t dstep){
	maxDiffs = new float[num_points];
	std::vector<std::thread> threads;
	float partial_max[num_threads*dims];
	float partial_min[num_threads*dims];
	for(int t = 0; t< num_threads; t++)
		threads.push_back(std::thread(&RANK::thread_calculate_maxDiffs, this, t, data, dstep, partial_max + t*dims, partial_min + t*dims));
	for(auto& t: threads) t.join();
	threads.clear();
	float max[dims];
	float min[dims];
	std::fill_n(max, dims, -std::numeric_limits<float>::max());
	std::fill_n(min, dims, std::numeric_limits<float>::max());
	for(int t = 0; t < num_threads; t++){
		for(int d = 0; d < dims; d++){
			if(max[d]<partial_max[t*dims + d]) max[d] = partial_max[t*dims + d];
			if(min[d]>partial_min[t*dims + d]) min[d] = partial_min[t*dims + d];
		}
	}
	for(int d = 0; d<dims; d++){
		maxDiffs[d] = max[d] - min[d];
	}
}

void RANK::calculate_alpha(float *data, size_t dstep){
	float *distances = new float[num_threads];
	int *counts = new int[num_threads];
	std::vector<std::thread> threads;
	for(int t = 0; t< num_threads; t++)
		threads.push_back(std::thread(&RANK::thread_calculate_alpha, this, t, data, dstep, distances+t, counts+t));
	for(auto& t: threads) t.join();
	alpha = 0;
	long long int num_combinations = (long long int)num_points*(long long int)(num_points - 1)/2;
	for(int t = 0; t<num_threads; t++) alpha += (double)(distances[t]*(double)counts[t]/num_combinations);
	alpha = -log(0.5)/alpha;
	delete[] distances;
	delete[] counts;
}

void RANK::thread_calculate_alpha(int thread_id, float *data, size_t dstep, float *distances, int *counts){
	int steps[dims];
	std::memcpy(steps, featsteps, sizeof(int)*dims);
	*distances = 0;
	*counts = 0;
	for(int point1 = thread_id; point1<num_points - 1; point1+=num_threads){
		for(int point2 = point1+1; point2<num_points; point2++){
			(*counts)++;
			register float distance = 0;
			for(int d = sFeat[f]; d<dims; d+=steps[d]){
				register float diff = data[point1*dstep + d] - data[point2*dstep + d];
				distance += diff*diff;
			}
			(*distances) += sqrtf(distance);
		}
	}
	(*distances) = (*distances)/(*counts);
}

void RANK::thread_calculate_maxDiffs(int thread_id, float *data, size_t dstep, float *max, float *min){
	std::fill_n(max, dims, -std::numeric_limits<float>::max());
	std::fill_n(min, dims, std::numeric_limits<float>::max());
	for(int point = thread_id; point<num_points; point+= num_threads){
		for(int d = 0; d<dims; d++){
			if(max[d]<data[point*dstep + d]) max[d] = data[point*dstep + d];
			if(min[d]>data[point*dstep + d]) min[d] = data[point*dstep + d];	
		}
	}
}

RANK::~RANK(){
	delete[] featsteps;
	delete[] sFeat;
}
