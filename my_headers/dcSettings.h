#pragma once

class dcSettings
{
public:
	dcSettings(std::string file);
	int dims[3], blockSize[3];
	std::vector<std::string> blockNames;
	std::vector<std::vector<int> > blockPositions;
	std::vector<int> unifyLabels;
	std::string targetName, targetLabelsName;
	~dcSettings();

private:
	void extract_settings(std::string variable, std::string value);
};