#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "my_settings.h"

settings::settings(std::string file){
	std::ifstream readFile(file.c_str());
	std::string variable, value, line;
	while(getline(readFile,line)){
		std::stringstream iss(line);
		getline(iss, variable, ':');
		getline(iss, value, ';');
		if (variable=="#") continue;
		extract_settings(variable, value);
	}
	readFile.close();
}

void settings::extract_settings(std::string variable, std::string value){
	if(variable == "glamx_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glamx_radius.push_back(std::stoi(segment));
	}
	if(variable == "glamy_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glamy_radius.push_back(std::stoi(segment));
	}
	if(variable == "glamz_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glamz_radius.push_back(std::stoi(segment));
	}
	if(variable == "glcmx_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glcmx_radius.push_back(std::stoi(segment));
	}
	if(variable == "glcmy_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glcmy_radius.push_back(std::stoi(segment));
	}
	if(variable == "glcmz_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			glcmz_radius.push_back(std::stoi(segment));
	}
	if(variable == "rlmx_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			rlmx_radius.push_back(std::stoi(segment));
	}
	if(variable == "rlmy_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			rlmy_radius.push_back(std::stoi(segment));
	}
	if(variable == "rlmz_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			rlmz_radius.push_back(std::stoi(segment));
	}
	if(variable == "fosx_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			fosx_radius.push_back(std::stoi(segment));
	}
	if(variable == "fosy_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			fosy_radius.push_back(std::stoi(segment));
	}
	if(variable == "fosz_radius"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			fosz_radius.push_back(std::stoi(segment));
	}
	if(variable == "x_radius") x_radius = std::stoi(value);
	if(variable == "y_radius") y_radius = std::stoi(value);
	if(variable == "z_radius") z_radius = std::stoi(value);
	if(variable == "step_x") step_x = std::stoi(value);
	if(variable == "step_y") step_y = std::stoi(value);
	if(variable == "step_z") step_z = std::stoi(value);
	if(variable == "box_size_x") box_size_x = std::stoi(value);
	if(variable == "box_size_y") box_size_y = std::stoi(value);
	if(variable == "box_size_z") box_size_z = std::stoi(value);
	if(variable == "number_of_min_clusters") number_of_min_clusters = std::stoi(value);
	if(variable == "number_of_max_clusters") number_of_max_clusters = std::stoi(value);
	if(variable == "number_of_threads") number_of_threads = std::stoi(value);
	if(variable == "exp_steps") radious_step = std::stoi(value);
	if(variable == "Cut_border") Cut_border = std::stoi(value);
	if(variable == "Border_size") Border_size = std::stof(value);
	if(variable == "feature_calculator") feature_calculator = std::stoi(value);	
	if(variable == "contrast") contrast = std::stoi(value);
	if(variable == "dissimilarity") dissimilarity = std::stoi(value);
	if(variable == "homogeneity") homogeneity = std::stoi(value);
	if(variable == "second_moment") second_moment = std::stoi(value);
	if(variable == "entropy") entropy = std::stoi(value);
	if(variable == "correlation") correlation = std::stoi(value);
	if(variable == "clusterShade") clusterShade = std::stoi(value);
	if(variable == "clusterProm") clusterProm = std::stoi(value);
	if(variable == "maxProb") maxProb = std::stoi(value);
	if(variable == "feature_dimentions") f_dims = std::stoi(value);
	if(variable == "glcm") glcm = std::stoi(value);
	if(variable == "glam") glam = std::stoi(value);
	if(variable == "rlm") rlm = std::stoi(value);
	if(variable == "fos") fos = std::stoi(value);
	if(variable == "lbp") lbp = std::stoi(value);
	if(variable == "rescale_features") rescaleFeatures = std::stoi(value);
	if(variable == "Photo_path") Photo_path = value;
	if(variable == "Photo_name") Photo_name = value;
	if(variable == "Labels_Path") Labels_Path = value;
	if(variable == "Photo_format") Photo_format = value;
	if(variable == "PCA_Threshold") pca_threshold = std::stof(value);
	if(variable == "Results_file") result_file = value;
	if(variable == "MaskLabels"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			maskLabels.push_back(std::stoi(segment));
	}
}

settings::~settings(){
}