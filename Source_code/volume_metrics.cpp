#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "volume_metrics.h"

void add_neighbors(int ***neighbor_map, int dims[3], std::vector<int> voxel, std::vector<std::vector<int> >& neighbors_to_check){
	for(int i=voxel[0]-1; i<=voxel[0]+1;i++){
		for(int j=voxel[1]-1; j<=voxel[1]+1;j++){
			for(int k=voxel[2]-1; k<=voxel[2]+1;k++){
				if((i<0) || (i>=dims[0]) || (j<0) || (j>=dims[1]) || (k<0) || (k>=dims[2]) || (neighbor_map[i][j][k] == 1)) continue;
				neighbor_map[i][j][k] = 1;
				std::vector<int> neighbor {i, j, k};
				neighbors_to_check.push_back(neighbor);
			}
		}
	}

}

void count_volumes(const cv::Mat& _src, int dims[3],int  number_of_clusters, std::vector<int>& number_of_volumes, std::vector<std::vector <int> >& volume_size){
	int ***map;
	int n_voxels = dims[0]*dims[1]*dims[2], done = 0;
	std::cout << "\rk: " << ((float)done)/n_voxels << std::flush;
	map = (int ***)malloc(dims[0]*sizeof(int **));
	for(int i = 0; i < dims[0]; i++){
		map[i] = (int **)malloc(dims[1]*sizeof(int *));
		for(int j = 0; j < dims[1]; j++){
			map[i][j] = (int *)malloc(dims[2]*sizeof(int));
			for (int k = 0; k<dims[2]; k++)
				map[i][j][k] = 0;
		}
	}
	// std::cout << "Line 1" << std::endl;
	// const float* src = (float*)_src.data;
	// size_t step_j = _src.step[0]/sizeof(src[0]);
	// size_t step_k = _src.step[1];
	for(int i = 0; i< number_of_clusters; i++){
		number_of_volumes.push_back(0);
		std::vector<int> temp;
		volume_size.push_back(temp);
	}
	// std::cout << "Line 2" << std::endl;
	for(int k = 0; k<dims[2]; k++){
		for(int i = 0; i<dims[0]; i++){
			for(int j = 0; j<dims[1]; j++){
				if(map[i][j][k] == 1) continue;
				int v_size = 1;
				std::vector<int> voxel {i, j, k};
				int ***neighbor_map;
				neighbor_map = (int ***)malloc(dims[0]*sizeof(int **));
				// std::cout << "Line 3" << std::endl;
				map[i][j][k] = 1;
				for (int x = 0; x<dims[0]; x++){
					neighbor_map[x] = (int **)malloc(dims[1]*sizeof(int *));
					for (int y = 0; y<dims[1]; y++){
						neighbor_map[x][y] = (int *)malloc(dims[2]*sizeof(int));
						for (int z = 0; z<dims[2]; z++){
							neighbor_map[x][y][z] = map[x][y][z];
						}
					}
				}
				std::cout << "\rk: " << ((float)done)/n_voxels << std::flush;
				// std::cout << "Line 4" << std::endl;
				int cluster = _src.at<int>(i, j, k);
				std::vector<std::vector<int> > neighbors_to_check;
				add_neighbors(neighbor_map, dims, voxel, neighbors_to_check);
				while(neighbors_to_check.size()>0){
					voxel = neighbors_to_check[0];
					if(_src.at<int>(voxel[0], voxel[1], voxel[2]) == cluster){
						map[voxel[0]][voxel[1]][voxel[2]] = 1;
						v_size++;
						done++;
						add_neighbors(neighbor_map, dims, voxel, neighbors_to_check);
					}
					neighbors_to_check.erase(neighbors_to_check.begin());
				}
				number_of_volumes[cluster]++;
				volume_size[cluster].push_back(v_size);
				for(int x = 0; x < dims[0]; x++){
					for(int y = 0; y < dims[1]; y++){
						free(neighbor_map[x][y]);
					}
					free(neighbor_map[x]);
				}
				free(neighbor_map);
			}
		}
	}
	// std::cout << "Line 5" << std::endl;
	for(int x = 0; x < dims[0]; x++){
		for(int y = 0; y < dims[1]; y++){
			free(map[x][y]);
		}
		free(map[x]);
	}
	free(map);
}

void from_clusterByColor_to_cluster(const std::vector<cv::Mat>& _src, int dims[3], cv::Mat& dst, int number_of_clusters){
	dst = cv::Mat(3, dims, CV_32S, cv::Scalar(0));
	int step = 256/(number_of_clusters+2);
	for(int i = 0; i<dims[0]; i++){
		for(int j = 0; j<dims[1]; j++){
			for(int k = 0; k<dims[2]; k++){
				dst.at<int>(i, j, k) = (int)(_src[k].at<uchar>(i, j))/step - 1;
			}
		}
	}
}

void from_labels_to_cluster(const cv::Mat& _src, int dims[3], cv::Mat& dst, int number_of_clusters){
	dst = cv::Mat(3, dims, CV_32S, cv::Scalar(0));
	for(int i = 0; i<dims[0]; i++){
		for(int j = 0; j<dims[1]; j++){
			for(int k = 0; i<dims[2]; k++){
				int voxel = dims[0]*dims[1]*k+dims[0]*j+i;
				dst.at<int>(i, j, k) = _src.at<int>(voxel);
			}
		}
	}
}

