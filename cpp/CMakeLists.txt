cmake_minimum_required(VERSION 3.18)
project(mcop_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Force pybind11 to use modern FindPython (so Python_EXECUTABLE is honored)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(_mcop_cpp
  src/bindings.cpp
  src/lsm.cpp
)

# Homebrew Eigen include path
target_include_directories(_mcop_cpp PRIVATE /opt/homebrew/opt/eigen/include/eigen3)
target_link_libraries(_mcop_cpp PRIVATE Eigen3::Eigen)

# Put the built module into the Python package folder
set_target_properties(_mcop_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/mcop"
)
