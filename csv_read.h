#ifndef CSV_READ_H
#define CSV_READ_H

#include <pybind11/numpy.h>
#include <string>

pybind11::array_t<float> read_data(const std::string& fname);

#endif  // CSV_READ_H
