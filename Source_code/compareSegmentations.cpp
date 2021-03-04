#include <stdio.h>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <thread>
#include <math.h>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "my_settings.h"
#include "dcSettings.h"

void thread_sucRate(int thread_id, int num_threads, int *predictions, int *labels, int num_points, unsigned long *counts){
	register int map[2][2] = {{1, 0}, {0, 1}};
	for(int i = thread_id; i < num_points; i+=num_threads){
		register int prediction_1 = predictions[i];
		register int label_1 = labels[i];
		for(int j = i+1; j<num_points; j++){
			counts[map[prediction_1 == predictions[j]][label_1 == labels[j]]]++;
		}
	}
}

float succes_rate(cv::Mat _predictions, cv::Mat _labels, int num_threads){
	unsigned long count[2] = {0, 0};
	if(_predictions.rows != _labels.rows){
		std::cout << "Number of predictions don't match with number of labels\nExiting.." << std::endl;
		exit(0);
	}
	int *labels = (int *)_labels.data;
	int *predictions = (int *)_predictions.data;
	std::vector<std::thread> threads;
	std::vector<unsigned long *> par_counts;
	for(int t=0; t<num_threads; t++){
		unsigned long *temp = (unsigned long *)malloc(2*sizeof(unsigned long));
		temp[0] = 0; temp[1] = 0;
		par_counts.push_back(temp);
		threads.push_back(std::thread(&thread_sucRate, t, num_threads ,predictions, labels, _predictions.rows, par_counts[t]));
	}
	for (auto& t: threads) t.join();
	std::cout << "I m here" << std::endl;
	for(int t=0; t< num_threads; t++){
		count[0] += par_counts[t][0];
		count[1] += par_counts[t][1];
		free(par_counts[t]);
	}
	double acc = ((double)count[1])/((double)(count[0]+count[1]));
	std::cout << count[0] << '-' << count[1] << '-' << count[1]+count[0] << '-' << ((unsigned long)_predictions.rows - 1)*((unsigned long)_predictions.rows - 1) << std::endl;
	std::cout << acc << std::endl;
	return (float)acc;
}

void makeMask(std::vector<int> &posLabels, int *labels, int numPoints, int *dst){
	for(int p = 0; p < numPoints; p++){
		if(std::find(posLabels.begin(), posLabels.end(), labels[p]) != posLabels.end())
			dst[p] = 1;
		else dst[p] = 0;
	}
}

void load_color_image(std::vector<cv::Mat> &image, int dim[3], int radius, std::string ph_name){
	std::string dir_name = ph_name + "/ClusterByColor";
	std::string extension = std::string(".bmp");
	for(int i = 0; i<dim[2] - 2*radius; i++){
		std::ostringstream is;
		is << i;
		std::string pic_name = dir_name + '/' + is.str() + extension;
		cv::Mat tmp = cv::imread(pic_name);
		image.push_back(tmp.clone());
	}
}


cv::Mat* load_image(settings Set, int dim[3]){
	cv::Mat *images;
	if(Set.Photo_format == "raw"){
		dcSettings tSet(Set.Photo_path);
		std::string dirname = tSet.targetName;
		dim[0] = tSet.dims[0]*tSet.blockSize[0];
		dim[1] = tSet.dims[1]*tSet.blockSize[1];
		dim[2] = tSet.dims[2]*tSet.blockSize[2];
		FILE *fp = NULL;
		int framesize = dim[0]*dim[1]*dim[2];
		unsigned char *imagedata = NULL;
		fp = fopen(dirname.c_str(), "rb");
		imagedata = (unsigned char*) malloc (sizeof(unsigned char) * framesize);
		fread(imagedata, sizeof(unsigned char), framesize, fp);
		fclose(fp);
		images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
		memcpy(images->data, imagedata, framesize);
		free(imagedata);
	}
	else if(Set.Photo_format == "rawb"){
		std::string suffix = std::string(".rawb");
		std::string dirname = Set.Photo_path + suffix;
		dim[0] = 181;
		dim[1] = 217;
		dim[2] = 181;
		FILE *fp = NULL;
		unsigned char *imagedata = NULL;
		int data_length = dim[0]*dim[1]*dim[2];
		fp = fopen(dirname.c_str(), "rb");
		imagedata = (unsigned char*) malloc (sizeof(unsigned char) * data_length);
		fread(imagedata, sizeof(unsigned char), data_length, fp);
		fclose(fp);
		cv::Mat img;
		images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
		for(int k = 0; k<dim[2]; k++){
			for(int i = 0; i<dim[0]; i++){
				for(int j = 0; j<dim[1]; j++){
					int pixelScalar = dim[0]*dim[1]*k + dim[0]*j+i;
					images->at<uchar>(i,j,k) = imagedata[pixelScalar];
				}
			}
		}
		free(imagedata);
	}
	else if(Set.Photo_format == "img"){
		std::string suffix = std::string(".img");
		std::string dirname = Set.Photo_path + suffix;
		dim[0] = 256;
		dim[1] = 256;
		dim[2] = 128;
		FILE *fp = NULL;
		short int *imagedata = NULL;
		int data_length = dim[0]*dim[1]*dim[2];
		fp = fopen(dirname.c_str(), "rb");
		imagedata = (short int*) malloc (sizeof(short int) * data_length);
		fread(imagedata, sizeof(short int), data_length, fp);
		fclose(fp);
		cv::Mat img;
		images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
		for(int k = 0; k<dim[2]; k++){
			for(int i = 0; i<dim[0]; i++){
				for(int j = 0; j<dim[1]; j++){
					int pixelScalar = dim[0]*dim[1]*k + dim[0]*j+i;
					images->at<uchar>(i,j,k) = (unsigned char)imagedata[pixelScalar];
				}
			}
		}
		free(imagedata);
	}
	else{
		std::cout << "Unrecognised photo format, exiting.." << std::endl;
		exit(0);
	}
	return images;
}


int main(int argc, char** argv){
	std::string ttt = std::string(argv[1]);
	settings Set(ttt);
	int dim[3];
	cv::Mat *images = load_image(Set, dim);
	// acimages = addBoarder(orImages, dims, min_radius);
	// images = &acimages;
	std::cout << "Siga mi tupotho" << std::endl;

	cv::Mat or_labels = cv::Mat(dim[0]*dim[1]*dim[2], 1, CV_32S);
	int *orLab = (int*)or_labels.data;
	if(Set.Photo_format == "img"){
		FILE *fp = NULL;
		fp = fopen(Set.Labels_Path.c_str(), "rb");
		short int *imagedata = (short int*) malloc (sizeof(short int) * dim[0]*dim[1]*dim[2]);
		fread(imagedata, sizeof(short int), dim[0]*dim[1]*dim[2], fp);
		fclose(fp);
		for(int point = 0; point < or_labels.rows; point++) orLab[point] = (int)((unsigned char)(imagedata[point]));
		free(imagedata);
	}
	else if(Set.Photo_format == "rawb"){
		FILE *fp = NULL;
		fp = fopen(Set.Labels_Path.c_str(), "rb");
		unsigned char *imagedata = (unsigned char*) malloc (sizeof(unsigned char) * dim[0]*dim[1]*dim[2]);
		fread(imagedata, sizeof(unsigned char), dim[0]*dim[1]*dim[2], fp);
		fclose(fp);
		for(int point = 0; point < or_labels.rows; point++) orLab[point] = (int)imagedata[point];		
		free(imagedata);
	}
	else if(Set.Photo_format == "raw"){
		dcSettings tSet(Set.Photo_path);
		FILE *fp = NULL;
		fp = fopen(tSet.targetLabelsName.c_str(), "rb");
		unsigned char *imagedata = (unsigned char*) malloc (sizeof(unsigned char) * dim[0]*dim[1]*dim[2]);
		fread(imagedata, sizeof(unsigned char), dim[0]*dim[1]*dim[2], fp);
		fclose(fp);
		for(int k = 0; k<dim[2]; k++){
			for(int i = 0; i<dim[0]; i++){
				for(int j = 0; j<dim[1]; j++){
					int pixelScalar = dim[0]*dim[1]*k + dim[0]*j+i;
					int cvgamotoxristosou = dim[2]*dim[1]*i + dim[2]*j+k;
					orLab[pixelScalar] = (int)imagedata[cvgamotoxristosou];
				}
			}
		}
		free(imagedata);
	}
	int *mask = new int[or_labels.rows];
	makeMask(Set.maskLabels, orLab, or_labels.rows, mask);

	cv::Mat maskLabels = cv::Mat(0, 1, CV_32S);
	for(int k = 0; k < dim[2]; k++){
		for(int j = 0; j < dim[1]; j++){
			for(int i = 0; i < dim[0]; i++){
				int pixelScalar = dim[0]*dim[1]*k + dim[0]*j+i;
				if(mask[pixelScalar]) maskLabels.push_back(or_labels.row(pixelScalar));
			}
		}
	}

	std::vector<cv::Mat> segmentations;
	std::vector<float> results;
	std::vector<std::string> results_methods;
	for(int method = 2; method < argc; method++){
		std::vector<cv::Mat> ClusterByColor;
		std::string ph_name = std::string(argv[method]);
		load_color_image(ClusterByColor, dim, 0, ph_name);
		int numVoxels = 0;
		for(int x = 0; x < dim[0]; x++)
			for(int y = 0; y < dim[1]; y++)
				for(int z = 0; z < dim[2]; z++)
					if(ClusterByColor[z].at<uchar>(x,y, 0) != 0) numVoxels++;
		std::cout << numVoxels << std::endl;
		cv::Mat lab = cv::Mat(numVoxels, 1, CV_32S, cv::Scalar(0));
		int *labels = (int *)lab.data;
		int pixelScalar = 0;
		for(int z = 0; z < dim[2]; z++){
			for(int y = 0; y < dim[1]; y++){
				for(int x = 0; x < dim[0]; x++){
					if(ClusterByColor[z].at<uchar>(x,y, 0) != 0){
						labels[pixelScalar] = (int)ClusterByColor[z].at<uchar>(x,y, 0);
						pixelScalar++;
					}
				}
			}
		}
		segmentations.push_back(lab);
	}
	float suc_rate = succes_rate(segmentations[0], maskLabels, 32);
	std::cout << "Success Rate = " << suc_rate << std::endl;
	// int numMethods = argc - 2;
	// for(int method = 0; method < numMethods - 1; method++){
	// 	for(int method2 = method + 1; method2 < numMethods; method2++){
	// 		std::cout << "Method 1 = " << method << ", Method 2 = " << method2 << std::endl;
	// 		float suc_rate = succes_rate(segmentations[method], segmentations[method2], 32);
	// 		std::cout << "Success Rate = " << suc_rate << std::endl;
	// 		results.push_back(suc_rate);
	// 		std::string ttt = std::string(argv[method2 + 2]);
	// 		std::string temp = std::string(argv[method + 2])+"|"+ttt;
	// 		results_methods.push_back(temp);
	// 	}
	// }
	// int k;
	// std::cin >> k;
	// for(int comb = 0; comb < results.size(); comb++){
	// 	std::string suc = std::to_string(results[comb]);
	// 	std::string photo = std::string(argv[1]);
	// 	std::string file_name = photo+std::string("_segmentation_comparison.txt");
	// 	std::ofstream out;
	// 	out.open(file_name, std::ios::app);
	// 	out << results_methods[comb] << ":" << suc << '\n';
	// 	out.close();
	// }
}