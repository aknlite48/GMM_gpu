#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "kernels.h"
#include "csv_read.h"

namespace py = pybind11;

void gmm_training(py::array_t<float> pi_k, py::array_t<float> u_k, py::array_t<float> E_k,
                  py::array_t<float> data, int K, int D, int N, float threshold) {
    auto pi_k_ptr = pi_k.mutable_data();
    auto u_k_ptr = u_k.mutable_data();
    auto E_k_ptr = E_k.mutable_data();
    auto data_ptr = data.mutable_data();

    GMM_training(pi_k_ptr, u_k_ptr, E_k_ptr, data_ptr, K, D, N, threshold);
}

py::array_t<int> gmm_inference(py::array_t<float> pi_k, py::array_t<float> u_k, py::array_t<float> E_k,
                               py::array_t<float> data, int K, int D, int N) {
    auto pi_k_ptr = pi_k.mutable_data();
    auto u_k_ptr = u_k.mutable_data();
    auto E_k_ptr = E_k.mutable_data();
    auto data_ptr = data.mutable_data();

    // Allocate output array for labels
    py::array_t<int> labels(N);
    auto labels_ptr = labels.mutable_data();

    GMM_inference(labels_ptr, pi_k_ptr, u_k_ptr, E_k_ptr, data_ptr, K, D, N);

    return labels;
}

// PyBind11 Module Definition
PYBIND11_MODULE(gmm_module, m) {
    m.doc() = "GMM Training and Inference with CUDA";

    m.def("gmm_training", &gmm_training, "Train the GMM model",
          py::arg("pi_k"), py::arg("u_k"), py::arg("E_k"), py::arg("data"),
          py::arg("K"), py::arg("D"), py::arg("N"), py::arg("threshold"));

    m.def("gmm_inference", &gmm_inference, "Run GMM inference",
          py::arg("pi_k"), py::arg("u_k"), py::arg("E_k"), py::arg("data"),
          py::arg("K"), py::arg("D"), py::arg("N"));

    m.def("read_data", &read_data, "Read a CSV file into a NumPy array",
        py::arg("filename"));

}
