#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "lsm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_mcop_cpp, m) {
    m.doc() = "C++ core for Monte Carlo American option pricing (LSM)";

    m.def("lsm_price_from_paths",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> paths,
             double K, double r, double T, bool is_call, int degree) {
              auto buf = paths.request();
              if (buf.ndim != 2) throw std::runtime_error("paths must be a 2D array");
              const int n_paths = static_cast<int>(buf.shape[0]);
              const int n_cols  = static_cast<int>(buf.shape[1]);
              if (n_cols < 2) throw std::runtime_error("paths must have at least 2 columns");

              const int n_steps = n_cols - 1;
              const double* ptr = static_cast<const double*>(buf.ptr);

              return lsm_price_from_paths(ptr, n_paths, n_steps, K, r, T, is_call, degree);
          },
          py::arg("paths"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("is_call"), py::arg("degree") = 2);
}
