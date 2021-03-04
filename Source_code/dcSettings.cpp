#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "dcSettings.h"

dcSettings::dcSettings(std::string file){
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

void dcSettings::extract_settings(std::string variable, std::string value){
	if(variable == "dims"){
		std::stringstream iss(value);
		std::string segment;
		int pos = 0;
		while(getline(iss, segment, ',')){
			if(pos > 2){
				std::cout << "Terribly wrong settings file" << std::endl;
				exit(0);
			}
			dims[pos] = std::stoi(segment);
			pos++;
		}
	}
	else if(variable == "blockDims"){
		std::stringstream iss(value);
		std::string segment;
		int pos = 0;
		while(getline(iss, segment, ',')){
			if(pos > 2){
				std::cout << "Terribly wrong settings file" << std::endl;
				exit(0);
			}
			blockSize[pos] = std::stoi(segment);
			pos++;
		}		
	}
	else if(variable == "targetName") targetName = value;
	else if(variable == "targetLabelsName") targetLabelsName = value;
	else if(variable == "UnifyLabels"){
		std::stringstream iss(value);
		std::string segment;
		while(getline(iss, segment, ','))
			unifyLabels.push_back(std::stoi(segment));
	}
	else{
		blockNames.push_back(variable);
		std::stringstream iss(value);
		std::string segment;
		std::vector<int> temp;
		while(getline(iss, segment, ','))
			temp.push_back(std::stoi(segment));
		blockPositions.push_back(temp);
	}
}

dcSettings::~dcSettings(){
}