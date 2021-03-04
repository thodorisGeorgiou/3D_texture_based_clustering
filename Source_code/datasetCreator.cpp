#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <sstream>
#include <stdio.h>
#include <pthread.h>
#include <limits>
#include <glob.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "dcSettings.h"

void load_block(cv::Mat &dst, std::string path, int dim[3]){
	FILE *fp = NULL;
	unsigned char *imagedata = NULL;
	int IMAGE_WIDTH =  dim[1];
	int IMAGE_HEIGHT =  dim[0]*dim[2];
	int framesize = IMAGE_HEIGHT*IMAGE_WIDTH;
	fp = fopen(path.c_str(), "rb");
	imagedata = (unsigned char*) malloc (sizeof(unsigned char) * framesize);
	fread(imagedata, sizeof(unsigned char), framesize, fp);
	// cv::Mat img;
	// img.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
	// memcpy(img.data, imagedata, framesize);
	fclose(fp);
	dst = cv::Mat(3, dim, CV_8UC1, cv::Scalar(0));
	for(int k = 0; k<dim[2]; k++){
		for(int i = 0; i<dim[0]; i++){
			for(int j = 0; j<dim[1]; j++){
				int pixelScalar = dim[0]*dim[1]*k + dim[0]*j+i;
				// dst.at<uchar>(i,j,k) = img.at<uchar>(i+k*dim[0], j);
				dst.at<uchar>(i,j,k) = imagedata[pixelScalar];
			}
		}
	}
	free(imagedata);
}


int main(int argc, char ** argv){
	std::string ttt = std::string(argv[1]);
	dcSettings Set(ttt);
	int targetDims[3] = {Set.dims[0]*Set.blockSize[0], Set.dims[1]*Set.blockSize[1], Set.dims[2]*Set.blockSize[2]};
	cv::Mat target = cv::Mat(3, targetDims, CV_8UC1, cv::Scalar(0));
	cv::Mat targetLabels = cv::Mat(3, targetDims, CV_8UC1, cv::Scalar(0));
	for(int photo = 0; photo < Set.blockNames.size(); photo++){
		cv::Mat block;
		load_block(block, Set.blockNames[photo], Set.blockSize);
		for(int position = 0; position < Set.blockPositions[photo].size(); position++){
			int mod = Set.blockPositions[photo][position] % (Set.dims[0]*Set.dims[1]);
			int k = Set.blockPositions[photo][position] / (Set.dims[0]*Set.dims[1]);
			int j = mod / Set.dims[0];
			int i = mod % Set.dims[0];
			std::cout << photo << ", k = " << k << ", j = " << j << ", i = " << i << ", position = " << Set.blockPositions[photo][position] << std::endl;
			for(int z = 0; z < Set.blockSize[2]; z++){
				for(int y = 0; y < Set.blockSize[1]; y++){
					for(int x = 0; x < Set.blockSize[0]; x++){
						int map[3] = {x+i*Set.blockSize[0], y+j*Set.blockSize[1], z+k*Set.blockSize[2]};
						target.at<uchar>(map[0], map[1], map[2]) = block.at<uchar>(x, y, z);
						targetLabels.at<uchar>(map[0], map[1], map[2]) = (unsigned char)photo;
					}
				}
			}
		}
	}
	int nElements = targetDims[0]*targetDims[1]*targetDims[2];
	FILE *out = fopen(Set.targetName.c_str(), "wb");
	fwrite(target.data, sizeof(unsigned char), nElements, out);
	fclose(out);
	out = fopen(Set.targetLabelsName.c_str(), "wb");
	fwrite(targetLabels.data, sizeof(unsigned char), nElements, out);
	fclose(out);
}

