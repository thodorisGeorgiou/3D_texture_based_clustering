#include <stdio.h>
#include <string>
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

#include "glcm3D.h"
#include "glam3D.h"
#include "rlm3D.h"
#include "fos3D.h"
#include "my_settings.h"
#include "dcSettings.h"
#include "my_kmeans.h"
#include "FuzzyCMeans.h"
#include "volume_metrics.h"
#include "mitra.h"
#include "mitra_var_1.h"
#include "mitra_var_2.h"
#include "clusterScatteringCriteria.h"
#include "rank.h"
#include "sRank.h"
#include "Irrelevancy_Filter.h"


cv::Mat random_samples(cv::Mat &data, float ratio){
	int n_samples = (int)(data.rows*ratio);
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, (double)data.rows);
	cv::Mat samples = cv::Mat(n_samples, data.cols, CV_32F);
	bool picked[data.rows];
	for(int i = 0; i< data.rows; i++) picked[i] = false;
	for(int s = 0; s<n_samples; s++){
		int sample = (int)dist(mt);
		if(picked[sample]){
			s--;
			continue;
		}
		picked[sample] = true;
		for(int f = 0; f<data.cols; f++) samples.at<float>(s, f) = data.at<float>(sample, f);
	}
	return samples;
}

void thread_rescaleFeatures(int thread_id, int num_threads, float *data, size_t dstep, int dims, int num_points, float *_max, float *_min){
	float max[dims], min[dims];
	std::memcpy(max, _max, dims*sizeof(float));
	std::memcpy(min, _min, dims*sizeof(float));
	for(int p = thread_id; p < num_points; p+=num_threads){
		register float *point = data + p*dstep;
		for(int f = 0; f<dims; f++)
			point[f] = (point[f] - min[f])/max[f];
	}
}

void thread_calculate_maxMin(int thread_id, int num_threads, float *data, size_t dstep, int dims, int num_points, float *max, float *min){
	std::fill_n(max, dims, -std::numeric_limits<float>::max());
	std::fill_n(min, dims, std::numeric_limits<float>::max());
	for(int p = thread_id; p<num_points; p+= num_threads){
		register float *point = data+p*dstep;
		for(int d = 0; d<dims; d++){
			if(max[d]<point[d]) max[d] = point[d];
			if(min[d]>point[d]) min[d] = point[d];	
		}
	}
}

void calculate_maxMin(float *data, size_t dstep, int num_points, int num_threads, int dims, float *max, float *min){
	std::vector<std::thread> threads;
	float partial_max[num_threads*dims];
	float partial_min[num_threads*dims];
	for(int t = 0; t< num_threads; t++)
		threads.push_back(std::thread(&thread_calculate_maxMin, t, num_threads, data, dstep, dims, num_points, partial_max + t*dims, partial_min + t*dims));
	for(auto& t: threads) t.join();
	threads.clear();
	std::fill_n(max, dims, -std::numeric_limits<float>::max());
	std::fill_n(min, dims, std::numeric_limits<float>::max());
	for(int t = 0; t < num_threads; t++){
		register int kapa = t*dims;
		for(int d = 0; d < dims; d++){
			if(max[d]<partial_max[kapa + d]) max[d] = partial_max[kapa + d];
			if(min[d]>partial_min[kapa + d]) min[d] = partial_min[kapa + d];
		}
	}
	for(int d = 0; d<dims; d++){
		if(max[d] == min[d]) max[d] = 1;
		else max[d] = max[d] - min[d];
	}
}



void rescaleFeatures(cv::Mat &features, int num_threads){
	float *max, *min;
	max = new float[features.cols];
	min = new float[features.cols];
	float *data = (float*)features.data;
	size_t step = features.step/sizeof(data[0]);
	calculate_maxMin(data, step, features.rows, num_threads, features.cols, max, min);
	std::vector<std::thread> threads;
	int num_points = features.rows;
	int dims = features.cols;
	for(int t = 0; t < num_threads; t++)
		threads.push_back(std::thread(&thread_rescaleFeatures, t, num_threads, data, step, dims, num_points, max, min));
	for(auto& t: threads) t.join();
	delete[] max;
	delete[] min;
}


void save_images(std::vector<std::vector<cv::Mat> > clusters,int radius,int dim[3], int num_clusters, std::string ph_name){
	mkdir(ph_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	std::string extension = std::string(".bmp");
	for(int i = 0; i<num_clusters; i++){
		std::ostringstream os;
		os << i;
		std::string dir_name = ph_name + "/Cluster_" + os.str();
		mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		for(int j = 0; j<dim[2]-2*radius; j++){
			std::ostringstream is;
			is << j;
			std::string pic_name = dir_name + '/' + is.str() + extension;
			cv::imwrite(pic_name, clusters[i][j]);
		}
	}
}

void save_color_image(std::vector<cv::Mat> image, std::string ph_name){
	mkdir(ph_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	std::string dir_name = ph_name + "/ClusterByColor";
	mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	std::string extension = std::string(".bmp");
	for(int i = 0; i<image.size(); i++){
		std::ostringstream is;
		is << i;
		std::string pic_name = dir_name + '/' + is.str() + extension;
		cv::imwrite(pic_name, image[i]);		
	}
}


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

float succes_rate(cv::Mat _predictions, cv::Mat _labels, settings Set){
	unsigned long count[2] = {0, 0};
	if(_predictions.rows != _labels.rows){
		std::cout << "Number of predictions don't match with number of labels\nExiting.." << std::endl;
		exit(0);
	}
	int *labels = (int *)_labels.data;
	int *predictions = (int *)_predictions.data;
	std::vector<std::thread> threads;
	std::vector<unsigned long *> par_counts;
	for(int t=0; t<Set.number_of_threads; t++){
		unsigned long *temp = (unsigned long *)malloc(2*sizeof(unsigned long));
		temp[0] = 0; temp[1] = 0;
		par_counts.push_back(temp);
		threads.push_back(std::thread(&thread_sucRate, t, Set.number_of_threads ,predictions, labels, _predictions.rows, par_counts[t]));
	}
	for (auto& t: threads) t.join();
	std::cout << "I m here" << std::endl;
	for(int t=0; t< Set.number_of_threads; t++){
		count[0] += par_counts[t][0];
		count[1] += par_counts[t][1];
		free(par_counts[t]);
	}
	double acc = ((double)count[1])/((double)(count[0]+count[1]));
	std::cout << count[0] << '-' << count[1] << '-' << count[1]+count[0] << '-' << ((unsigned long)_predictions.rows - 1)*((unsigned long)_predictions.rows - 1) << std::endl;
	std::cout << acc << std::endl;
	return (float)acc;
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
	else if(Set.Photo_format == "jpg"){
		std::string dirname = Set.Photo_path;
		std::string suffix = ".jpg";
		int bord_length = 0;
		cv::Mat tmp = cv::imread("T24_new/T24_R1_1.jpg");
		dim[0] = tmp.rows;
		dim[1] = tmp.cols;
		dim[2] = 479;
		if(Set.Cut_border == 1) bord_length = dim[1]*Set.Border_size;
		dim[1] -= 2*bord_length;
		images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
		for (int k = 1; k < dim[2]+1; k++){
			std::ostringstream os;
			os << k;
			std::string dir_name = dirname + os.str() + suffix;
			cv::Mat temp = cv::imread(dir_name, 0);
			for (int i = 0; i < dim[0]; i++){
				for (int j = 0; j<dim[1]; j++){
					images->at<uchar>(i, j, k - 1) = temp.at<uchar>(i,j+bord_length);
				}
			}
		}
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

cv::Mat join_features(std::vector<cv::Mat> features, std::vector<int> dimensions, int n_pixels){
	int n_features = 0;
	for(std::vector<int>::iterator j=dimensions.begin();j!=dimensions.end();++j) n_features += *j;
	int sizes[2] = {n_pixels, n_features};
	cv::Mat temp(2, sizes, CV_32F, cv::Scalar(0));
	int offset = 0;
	for(int j = 0; j < features.size(); j++){
		for(int i = 0; i < n_pixels; i++){
			for(int k = 0; k < dimensions[j]; k++){
				temp.at<float>(i, k + offset) = features[j].at<float>(i, k);
			}
		}
		offset += dimensions[j];
		features[j].release();
	}
	return temp;
}

void makeMask(std::vector<int> &posLabels, int *labels, int numPoints, int *dst){
	for(int p = 0; p < numPoints; p++){
		if(std::find(posLabels.begin(), posLabels.end(), labels[p]) != posLabels.end())
			dst[p] = 1;
		else dst[p] = 0;
	}
}

void features2MaskFeatures(cv::Mat &features, cv::Mat &dst, int *mask, std::vector<int> &map){
	dst = cv::Mat(0, features.cols, CV_32F);
	for(int p = 0; p < features.rows; p++){
		if(mask[p]){
			map.push_back(p);
			dst.push_back(features.row(p));
		}
	}
}

int main(int argc, char** argv )
{
	if(argc != 2){
		std::cout << "Wrong number of input files, please include only the settings file.\nExiting.." << std::endl;
		exit(0);
	}
	std::string ttt = std::string(argv[1]);
	settings Set(ttt);
	int pixel, max_radius, min_radius;
	int dim[3];
	cv::Mat *images;
	max_radius = Set.x_radius;
	min_radius = Set.y_radius;
	images = load_image(Set, dim);
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
					int inverseScalar = dim[2]*dim[1]*i + dim[2]*j+k;
					if(std::find(tSet.unifyLabels.begin(), tSet.unifyLabels.end(), (int)imagedata[inverseScalar]) != tSet.unifyLabels.end())
						orLab[pixelScalar] = tSet.unifyLabels[0];
					else
						orLab[pixelScalar] = (int)imagedata[inverseScalar];
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

	int n_features = 0;
	time_t start_time = time(0);
	std::vector<cv::Mat> features;
	std::vector<int> dimensions;
	cv::Mat testing_glcm_features, testing_rlm_features, testing_glam_features, testing_fos_features;
	if(Set.glcm == 1){
		for(int rdInd = 0; rdInd < Set.glcmx_radius.size(); rdInd++){
			if(Set.feature_calculator == 0)
				dimensions.push_back(8*12);
			else if (Set.feature_calculator == 1)
				dimensions.push_back(15*12);
			else{
				std::cout << "Wrong ferature calculator, exiting.." << std::endl;
				exit(0);
			}				
			n_features += dimensions.back();
			GLCM_3d glcm(Set.glcmx_radius[rdInd], Set.glcmy_radius[rdInd], Set.glcmz_radius[rdInd], Set.step_x, Set.step_y, Set.step_z, 64, Set.number_of_threads, Set.feature_calculator);
			std::cout << "Start computing GLCM 3d" << std::endl;
			glcm.computeFeatures(images, dim, mask);
			std::cout << "Done!!" << std::endl;
			testing_glcm_features = glcm.getFeatures();
			features.push_back(testing_glcm_features.clone());
		}
	}
	if(Set.glam == 1){
		for(int rdInd = 0; rdInd < Set.glamx_radius.size(); rdInd++){
			if(Set.feature_calculator == 0)
				dimensions.push_back(8);
			else if(Set.feature_calculator == 1)
				dimensions.push_back(15);
			else{
				std::cout << "Wrong feature calculator, exiting.." << std::endl;
				exit(0);
			}
			n_features += dimensions.back();
			GLAM_3d glam(Set.glamx_radius[rdInd], Set.glamy_radius[rdInd], Set.glamz_radius[rdInd], Set.box_size_x, Set.box_size_y, Set.box_size_z, 64,
				Set.number_of_threads, Set.feature_calculator);
			std::cout << "Start computing GLAM 3d" << std::endl;
			glam.computeFeatures(images, dim, mask);
			testing_glam_features = glam.getFeatures();
			std::cout << "Done!!" << std::endl;
			features.push_back(testing_glam_features.clone());
		}
	}
	if(Set.rlm == 1){
		for(int rdInd = 0; rdInd < Set.rlmx_radius.size(); rdInd++){
			dimensions.push_back(5*12);
			n_features += dimensions.back();
			RLM_3d rlm(Set.rlmx_radius[rdInd], Set.rlmy_radius[rdInd], Set.rlmz_radius[rdInd], Set.step_x, Set.step_x, Set.step_x, 64, Set.number_of_threads);
			std::cout << "Start computing RLM 3d" << std::endl;
			rlm.computeFeatures(images, dim, mask);
			testing_rlm_features = rlm.getFeatures();
			std::cout << "Done!!" << std::endl;
			features.push_back(testing_rlm_features.clone());
		}
	}
	if(Set.fos == 1){
		for(int rdInd = 0; rdInd < Set.fosx_radius.size(); rdInd++){
			dimensions.push_back(5);
			n_features += dimensions.back();
			FOS fos(Set.fosx_radius[rdInd], Set.fosy_radius[rdInd], Set.fosz_radius[rdInd], Set.step_x, Set.step_x, Set.step_x, 256);
			std::cout << "Start computing FOS 3d" << std::endl;
			fos.computeFeatures(images, dim, mask);
			testing_fos_features = fos.getFeatures();
			std::cout << "Done!!" << std::endl;
			features.push_back(testing_fos_features.clone());
		}
	}

	double elapsed = difftime( time(0), start_time);
	std::cout << "Time for feature extraction: " << elapsed << std::endl;

	int n_pixels = maskLabels.rows;
	cv::Mat voxel_features = join_features(features, dimensions, n_pixels);

	if(Set.rescaleFeatures == 1) rescaleFeatures(voxel_features, Set.number_of_threads);

	MITRA_VAR_1 feat_extractor = MITRA_VAR_1(Set.number_of_threads, (float)0.15);
	n_features = feat_extractor.select_features(voxel_features);

	IFilter nfs = IFilter(Set.number_of_threads, 0.1, voxel_features.cols, 0.005, 50);
	nfs.select_features(voxel_features);
	float *weights = nfs.getWeights();
	nfs.transorm_data(voxel_features);

	for(int n_clusters = Set.number_of_min_clusters; n_clusters <= Set.number_of_max_clusters; n_clusters++){
		cv::Mat tlabels(voxel_features.rows, 1, CV_32S), centers(n_clusters, n_features, CV_32F);
		std::cout << "Clustering for " << n_clusters << " clusters..." << std::endl;
		ParrallelKMeans KMeans(Set.number_of_threads, 1500, 0.00001, n_features, 30);
		double lamda = KMeans.cluster(voxel_features, n_clusters, centers, tlabels);
		std::cout << "Starting FCM" << std::endl;
		FuzzyCMeans fcm(Set.number_of_threads, 0.00001, n_clusters, n_features, voxel_features.rows, 2);
		fcm.cluster(voxel_features, centers);
		cv::Mat tfcm_labels = fcm.getLabels();
		std::cout << "Done Clustering!! " << std::endl;
		std::vector<cv::Mat> ClusterByColor;
		std::vector<std::vector<cv::Mat> > clusters;
		std::vector<cv::Mat> ClusterByColorFCM;
		std::vector<std::vector<cv::Mat> > clustersFCM;
		for (int i = 0; i<n_clusters; i++)
		{
			std::vector<cv::Mat> tmp_cluster;
			std::vector<cv::Mat> tmp_cluster2;
			for (int z = 0; z < dim[2]; z++)
			{
				cv::Mat tmp = cv::Mat::zeros(dim[0], dim[1], CV_8UC1);
				tmp_cluster.push_back(tmp.clone());
				tmp_cluster2.push_back(tmp.clone());
			}
			clusters.push_back(tmp_cluster);
			clustersFCM.push_back(tmp_cluster2);
		}
		std::vector<cv::Mat> sliced_images;
		std::vector<cv::Mat> sliced_imagesFCM;
		int index = 0;
		int step = 256/(n_clusters+2);
		for (int z = 0; z < dim[2]; z++){
			cv::Mat tmp  = cv::Mat::zeros(dim[0], dim[1], CV_8UC1);
			ClusterByColor.push_back(tmp.clone());
			ClusterByColorFCM.push_back(tmp.clone());
			for(int j=0;j<dim[1];j++){
				for(int i=0;i<dim[0];i++) {
					pixel = (dim[0])*(dim[1])*z + (dim[0])*j + i;
					if(mask[pixel]){
						clusters[tlabels.at<int>(index)][z].at<uchar>(i, j) = images->at<uchar>(i, j, z);
						ClusterByColor[z].at<uchar>(i, j) = (unsigned char)(step*(tlabels.at<int>(index) + 1));
						tmp.at<uchar>(i, j) = images->at<uchar>(i, j, z);
						clustersFCM[tfcm_labels.at<int>(index)][z].at<uchar>(i, j) = images->at<uchar>(i, j, z);
						ClusterByColorFCM[z].at<uchar>(i, j) = (unsigned char)(step*(tfcm_labels.at<int>(index) + 1));
						index++;
					}
				}
			}
			sliced_images.push_back(tmp);
		}
		std::string tname = Set.Photo_name + "/km";
		std::string clusterFolder = std::string("NFSelector");
		float tsuc_rate = succes_rate(maskLabels, tlabels, Set);
		std::cout << "KMeans Success Rate = " << tsuc_rate << std::endl;

		std::string fcmtname = Set.Photo_name + "/fcm";
		float fcmtsuc_rate = succes_rate(maskLabels, tfcm_labels, Set);
		std::cout << "FCM Success Rate = " << tsuc_rate << std::endl;
		std::string radious = std::to_string(Set.x_radius);
		mkdir(tname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		tname = tname + "/" + clusterFolder;
		mkdir(tname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		save_images(clusters, 0, dim, n_clusters, tname);
		save_color_image(ClusterByColor, tname);

		std::string suc = std::to_string(tsuc_rate);
		std::string file_name = Set.result_file + "_km.txt";
		std::ofstream out;
		out.open(file_name, std::ios::app);
		out <<  suc << '\n';
		out.close();

		mkdir(fcmtname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		fcmtname = fcmtname + "/" + clusterFolder;
		mkdir(fcmtname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		save_images(clustersFCM, 0, dim, n_clusters, fcmtname);
		save_color_image(ClusterByColorFCM, fcmtname);

		suc = std::to_string(fcmtsuc_rate);
		file_name = Set.result_file + "_fcm.txt";
		out.open(file_name, std::ios::app);
		out <<  suc << '\n';
		out.close();

	}

	return 0;
}