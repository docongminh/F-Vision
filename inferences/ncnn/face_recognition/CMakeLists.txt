#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(VisionEngine)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

message(${SRC_FILES})

# list(APPEND SRCS ${LAYER_ARCH_SRC})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11 -fopenmp")
add_library(VisionEngine STATIC ${SRC_FILES})

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )
include_directories(${CMAKE_SOURCE_DIR}/ncnn/include/ncnn)
link_directories(${CMAKE_SOURCE_DIR}/ncnn/lib)
target_link_libraries(VisionEngine PUBLIC ${OpenCV_LIBS} ncnn)
#
find_package(OpenMP)
if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
    target_compile_options(VisionEngine PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# if(MIRROR_OPENMP AND OpenMP_CXX_FOUND)
#     message("Building with OpenMP")
#     target_link_libraries(VisionEngine PUBLIC OpenMP::OpenMP_CXX)
# endif()

#
target_include_directories(VisionEngine
    PUBLIC
        $<INSTALL_INTERFACE:include/vision>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/recognition>
)
#
add_executable(masked_face ${CMAKE_SOURCE_DIR}/src/inference.cpp)
target_link_libraries(masked_face PUBLIC VisionEngine)
