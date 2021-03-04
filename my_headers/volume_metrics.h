
void add_neighbors(int ***neighbor_map, int dims[3], std::vector<int> voxel, std::vector<std::vector<int> >& neighbors_to_check);
void count_volumes(const cv::Mat& _src, int dims[3],int  number_of_clusters, std::vector<int>& number_of_volumes, std::vector<std::vector <int> >& volume_size);
void from_clusterByColor_to_cluster(const std::vector<cv::Mat>& _src, int dims[3], cv::Mat& dst, int number_of_clusters);
void from_labels_to_cluster(const cv::Mat& _src, int dims[3], cv::Mat& dst, int number_of_clusters);
