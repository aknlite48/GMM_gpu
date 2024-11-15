#include<iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void read_data(float* q,std::string fname) {
	std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }
    std::string line; int i=0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        //std::vector<float> row;

        while (std::getline(ss, item, ',')) {
            //row.push_back(std::stof(item)); 
            *(q+i)=std::stof(item);
        }
        q.push_back(row);
}
    file.close();
}