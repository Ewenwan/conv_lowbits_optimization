cmake_minimum_required(VERSION 2.8)
project(search_gemm)
find_package(CUDA REQUIRED)
include(FindCUDA)
set(CMAKE_BUILD_TYPE release)
include_directories(${CUDA_INCLUDE_DIRS})
set(testuf_SRCS 
	 main.cpp 
         util.h
	 )

cuda_add_executable(search_gemm ${testuf_SRCS})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)

target_link_libraries(search_gemm
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
