
cmake_minimum_required(VERSION 2.8)
project(test-nnpack)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE debug)

# try any external nnpack first
#find_package(NNPACK)
#if (NNPACK_FOUND)
#	message(INFO "Found external NNPACK installation.")
#	return()
#endif()
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_package(nnpack REQUIRED)
#set(NNPACK_INCLUDE_DIRS "~/NNPACK/build" "~/NNPACK/include" "~/NNPACK/nnpack")
#set(NNPACK_DIR "~/NNPACK/build")

include_directories(${NNPACK_INCLUDE_DIRS})
#include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCE_FILES
	        nnpack-main.cpp)

add_executable(test-nnpack ${SOURCE_FILES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")
#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)

target_link_libraries(test-nnpack 
	              /usr/local/lib/libnnpack.a 
		      /usr/local/lib/libpthreadpool.a 
		      pthread
		      cpuinfo)
