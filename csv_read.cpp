#include "csv_read.h"
#include <pybind11/numpy.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace py = pybind11;

py::array_t<float> read_data(const std::string& fname) {
    std::ifstream file(fname);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open the file: " + fname);
    }

    std::vector<float> data;  // Temporary vector to store the data
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        while (std::getline(ss, item, ',')) {
            data.push_back(std::stof(item));  // Convert to float and store in vector
        }
    }

    file.close();

    // Create a NumPy array from the vector
    ssize_t size = data.size();
    auto result = py::array_t<float>(size);  // Allocate NumPy array
    auto result_ptr = result.mutable_data(); // Pointer to NumPy array memory

    // Copy data from the vector to the NumPy array
    std::memcpy(result_ptr, data.data(), size * sizeof(float));

    return result;
}
