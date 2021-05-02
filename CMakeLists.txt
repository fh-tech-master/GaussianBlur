cmake_minimum_required(VERSION 3.17)
project(gaussian_blur)

set(CMAKE_CXX_STANDARD 14)

add_executable(gaussian_blur main.cpp gaussian_blur.h gaussian_blur.cpp tga.h tga.cpp cxxopts.hpp)