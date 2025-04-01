cmake_minimum_required(VERSION 3.22)

project(assignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(libraw CONFIG REQUIRED)

add_executable(assignment-1 main.cpp)
target_link_libraries(assignment-1 PRIVATE libraw::raw opencv_core opencv_photo opencv_highgui)