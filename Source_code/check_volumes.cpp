#include <string>
#include <string.h>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <limits>
#include <glob.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "my_settings.h"
#include "volume_metrics.h"

#include "vtkAutoInit.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyle.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkImageResample.h"
#include "vtkMetaImageReader.h"
#include "vtkProperty.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkCamera.h"
#include "vtkCommand.h"
#include "vtkSmartVolumeMapper.h"




#define VTI_FILETYPE 1
#define MHA_FILETYPE 2
#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)


void load_images(std::vector<std::vector<cv::Mat> > &clusters,int radius,int dim[3], int num_clusters, std::string ph_name){
	std::string extension = std::string(".bmp");
	for(int i = 0; i<num_clusters; i++){
		std::ostringstream os;
		os << i;
		std::string dir_name = ph_name + "/Cluster_" + os.str();
		std::vector<cv::Mat> temp_cluster;
		for(int j = 0; j<dim[2]-2*radius; j++){
			std::ostringstream is;
			is << j;
			std::string pic_name = dir_name + '/' + is.str() + extension;
			cv::Mat tmp = cv::imread(pic_name);
			temp_cluster.push_back(tmp.clone());
		}
		clusters.push_back(temp_cluster);
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



int main(int argc, char** argv ){
	if(argc < 2){
		std::cout << "Wrong input 1" << std::endl;
		exit(0);
	}
	std::vector<std::string> ttt;
	std::vector<settings> Set;
	for(int i = 1; i<argc; i++){
		std::string str = std::string(argv[i]);
		ttt.push_back(str);
		Set.push_back(ttt[i-1]);
	}
	// std::string ttt = std::string(argv[1]);
	// settings Set(ttt);
	for(int photo = 0; photo<argc-1; photo++){
		int pixel, x_radius, y_radius, z_radius;
		int dim[3];
		// cv::Mat *images;
		if (Set[photo].Photo_format == "dicom"){
			const char *dirname = Set[photo].Photo_path.c_str();
			vtkImageData *input=0;
			vtkDICOMImageReader *dicomReader = vtkDICOMImageReader::New();
		    dicomReader->SetDirectoryName(dirname);
		    dicomReader->Update();
		    input=dicomReader->GetOutput();
		    input->GetDimensions(dim);
			// images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
			// for (int z = 0; z < dim[2]; z++)
			// {
			// 	for (int y = 0; y < dim[1]; y++)
			// 	{
			// 		for (int x = 0; x < dim[0]; x++)
			// 		{
			// 			unsigned char* pixel = static_cast<unsigned char*>(input->GetScalarPointer(x,y,z));
			// 			images->at<uchar>(x, y, z) = pixel[0];
			// 		}
			// 	}
			// }
		}
		else if(Set[photo].Photo_format == "jpg"){
			std::string dirname = Set[photo].Photo_path;
			std::string suffix = ".jpg";
			cv::Mat tmp = cv::imread("T24_new/T24_R1_1.jpg");
			dim[0] = tmp.rows;
			dim[1] = tmp.cols;
			dim[2] = 479;
			int bord_length = 0;
			if(Set[photo].Cut_border == 1) bord_length = dim[1]*Set[photo].Border_size;
			dim[1] -= 2*bord_length;			
			// images = new cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
			// for (int k = 1; k < dim[2]+1; k++){
			// 	std::ostringstream os;
			// 	os << k;
			// 	std::string dir_name = dirname + os.str() + suffix;
			// 	cv::Mat temp = cv::imread(dir_name, 0);
			// 	for (int i = 0; i < dim[0]; i++){
			// 		for (int j = 0; j<dim[1]; j++){
			// 			images->at<uchar>(i, j, k - 1) = temp.at<uchar>(i,j);
			// 		}
			// 	}
			// }
		}
		else{
			std::cout << "Wrong input 2" << std::endl;
			exit(0);
		}
		std::vector<cv::Mat> ClusterByColor;
		cv::Mat image;
		// std::vector<std::vector<cv::Mat> > clusters;
		// load_images(clusters, z_radius, dim, Set[photo].number_of_clusters, Set[photo].Photo_name);
		x_radius = Set[photo].x_radius;
		y_radius = Set[photo].y_radius;
		z_radius = Set[photo].z_radius;
		int dims[3] = {dim[0] - 2*x_radius, dim[1] - 2*y_radius, dim[2] - 2*z_radius};
		std::vector<int> number_of_volumes;
		std::vector<std::vector <int> > volume_size;
		std::cout << "Start: " << Set[photo].Photo_name << std::endl;
		load_color_image(ClusterByColor, dim, z_radius , Set[photo].Photo_name);
		std::cout << "Photo Loaded" << std::endl;
		from_clusterByColor_to_cluster(ClusterByColor, dims, image, Set[photo].number_of_min_clusters);
		std::cout << "Photo translated.." << std::endl;
		count_volumes(image, dims, Set[photo].number_of_min_clusters, number_of_volumes, volume_size);
		std::cout << std::endl;
		// std::vector<int> total_volumes;
		// std::vector<std::vector<float> > volume_percetages;
		for(int c = 0; c<Set[photo].number_of_min_clusters; c++){
			std::cout << "Cluster " << c << ':' << number_of_volumes[c] << " Volumes" << std::endl;
			std::sort(volume_size[c].begin(), volume_size[c].end(), std::greater<int>());
			int temp_sum = 0;
			for(std::vector<int>::iterator p = volume_size[c].begin(); p != volume_size[c].end(); ++p){
				temp_sum += *p;
			}
			for(int p = 0; p < 20; p++){
				std::cout << ((float)volume_size[c][p])/((float)temp_sum) << " ";
				if((p+1) % 10 == 0) std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
	return 0;
}