cmake_minimum_required(VERSION 3.14)
project(improved_genetic_vrp_final)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

pybind11_add_module(improved_genetic_vrp_final improved_genetic_vrp_final.cpp)
