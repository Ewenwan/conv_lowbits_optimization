
cmake_minimum_required(VERSION 2.8)
project(test-gemmlowp)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE debug)

# try any external nnpack first
#find_package(NNPACK)
#if (NNPACK_FOUND)
#	message(INFO "Found external NNPACK installation.")
#	return()
#endif()
#SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
#find_package(test-gemmlowp REQUIRED)
#set(NNPACK_INCLUDE_DIRS "~/NNPACK/build" "~/NNPACK/include" "~/NNPACK/nnpack")
#set(NNPACK_DIR "~/NNPACK/build")

include_directories(${NNPACK_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/eight_bit_int_gemm)
include_directories(${CMAKE_SOURCE_DIR}/public)
include_directories(${CMAKE_SOURCE_DIR}/internal)
include_directories(${CMAKE_SOURCE_DIR}/profiling)
include_directories(${CMAKE_SOURCE_DIR}/fixedpoint)

set(eight_bit_int_gemm
	        eight_bit_int_gemm/eight_bit_int_gemm.cc
                eight_bit_int_gemm/eight_bit_int_gemm.h)

set(HEADER_FILES
	        public/gemmlowp.h
                public/output_stages.h)
set(INTERNAL_FILES
	        internal/allocator.h            
		internal/kernel_default.h    
		internal/multi_thread_gemm.h  
		internal/pack_neon.h                      
		internal/simd_wrappers_neon.h
		internal/block_params.h         
		internal/kernel.h            
		internal/output.h             
		internal/pack_sse.h                       
		internal/simd_wrappers_sse.h
		internal/common.h               
		internal/kernel_neon.h       
		internal/output_neon.h        
		internal/platform.h                       
		internal/single_thread_gemm.h
		internal/compute.h              
		internal/kernel_reference.h  
		internal/output_sse.h         
		internal/simd_wrappers_common_neon_sse.h  
		internal/unpack.h
		internal/dispatch_gemm_shape.h  
		internal/kernel_sse.h        
		internal/pack.h               
		internal/simd_wrappers.h
		)

set(PROFILE_FILES
	       profiling/instrumentation.h  
	       profiling/profiler.h  
	       profiling/pthread_everywhere.h
		)

set(FIXEDPOINT_FILES
		fixedpoint/fixedpoint.h  
		fixedpoint/fixedpoint_neon.h  
		fixedpoint/fixedpoint_sse.h)	
add_executable(test-gemmlowp quantization_example.cc ${eight_bit_int_gemm} ${HEADER_FILES} ${INTERNAL_FILES} ${PROFILE_FILES} ${FIXEDPOINT_FILES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -msse4.1 -lpthread -ffast-math -Wall")
#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)

target_link_libraries(test-gemmlowp
	              /usr/local/lib/libnnpack.a 
		      /usr/local/lib/libpthreadpool.a 
		      pthread
		      cpuinfo)
