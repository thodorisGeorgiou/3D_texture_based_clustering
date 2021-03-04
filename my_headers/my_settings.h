#pragma once

class settings
{
public:
	settings(std::string file);
	int x_radius;
	int y_radius;
	int z_radius;
	int step_x;
	int step_y;
	int step_z;
	int box_size_x;
	int box_size_y;
	int box_size_z;
	int number_of_min_clusters;
	int number_of_max_clusters;
	int number_of_threads;
	int radious_step;
	int feature_calculator;
	int contrast;
	int dissimilarity;
	int homogeneity;
	int second_moment;
	int Cut_border;
	float Border_size;
	float pca_threshold;
	int entropy;
	int correlation;
	int clusterShade;
	int clusterProm;
	int maxProb;
	int f_dims;
	int glcm;
	int glam;
	int rlm;
	int fos;
	int lbp;
	int rescaleFeatures;
	std::string Photo_path;
	std::string Labels_Path;
	std::string Photo_name;
	std::string Photo_format;
	std::string result_file;
	std::vector<int> maskLabels;
	std::vector<int> glcmx_radius;
	std::vector<int> glcmy_radius;
	std::vector<int> glcmz_radius;
	std::vector<int> glamx_radius;
	std::vector<int> glamy_radius;
	std::vector<int> glamz_radius;
	std::vector<int> rlmx_radius;
	std::vector<int> rlmy_radius;
	std::vector<int> rlmz_radius;
	std::vector<int> fosx_radius;
	std::vector<int> fosy_radius;
	std::vector<int> fosz_radius;
	~settings();

private:
	void extract_settings(std::string variable, std::string value);
};