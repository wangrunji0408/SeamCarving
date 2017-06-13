cmake_minimum_required(VERSION 3.7)
project(3SeamCarving)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
find_package( OpenCV REQUIRED )

set(SOURCE_FILES src/main.cpp src/MouseDraw.h)
add_executable(3SeamCarving ${SOURCE_FILES})
target_link_libraries( 3SeamCarving ${OpenCV_LIBS} )