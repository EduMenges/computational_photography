cmake_minimum_required(VERSION 3.22)

project(assignment-2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(assignment-2 main.cpp)
target_link_libraries(assignment-2 opencv_core opencv_photo opencv_highgui)