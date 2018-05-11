cmake_minimum_required(VERSION 2.8)
project(test)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE debug)

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.5")
include(FindCUDA)

find_package(CUDA REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS})

#add_executable(test cudnn_conv_float32.cc)
add_executable( test cudnn_conv_int8.cc)


LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall -fno-stack-protector")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)

target_link_libraries(test
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        libcudnn.so
        )

