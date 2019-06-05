cmake_minimum_required(VERSION 3.9)
project(edge_nms)

set(CMAKE_CXX_STANDARD 11)


# 设置输出路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(DIR_SRCS_ "./src")
# aux_source_directory(./src DIR_SRCS_)
# add_library(gradient2 SHARED ${DIR_SRCS_}/gradient2.h ${DIR_SRCS_}/gradientMex.cpp ${DIR_SRCS_}/sse.hpp ${DIR_SRCS_}/wrappers.hpp)
# add_library(convTriangle SHARED ${DIR_SRCS_}/convTri.cpp ${DIR_SRCS_}/convTri.h ${DIR_SRCS_}/sse.hpp ${DIR_SRCS_}/wrappers.hpp)
add_library(edges_nms SHARED ${DIR_SRCS_}/edgesNmsMex.cpp)
# add_library(eval_edge STATIC ${DIR_SRCS_})
