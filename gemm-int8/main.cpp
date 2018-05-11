#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <sys/time.h>
#include <time.h>
#include <fstream>

#include <vector>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <dirent.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "util.h"

using namespace std;

template <typename F>
double benchmark(const F &fun, int n = 10000)
{
    double duration = 0.0;
    std::vector<double> timings;
    for (int i = 0; i < n; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        fun();
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timings.push_back(duration);
    }

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}
void simple_sgemm(int M,int N, int K ,const int8_t *A, const int8_t *B, int8_t *C) {
    int i, j, k;
    for(i=0; i<M; i++)
    for(j=0; j<N; j++) {
        float s=0;
        for(k=0; k<K; k++) s+=A[k*K+i]*B[j*N+k];
        C[j*N+i]=s;
    }
}
/* Matrix size */
//#define M  (50)
//#define N  (25)
//#define K  (100)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, int alpha, const  int8_t *A, const int8_t *B,
                         int beta, int8_t *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            long prod = 0;

           for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

int main(int argc, char* argv[])
{
   cudaSetDevice(atoi(argv[1]));
   //int DIM = atoi(argv[2])   ;
   int face_num = 2*4096;//5120;//10240;//DIM;//4096;
   int database_num = 2*4096;//2*4096;//DIM;//4096;
   int feature_dim = 2*4096;//2*4096;//DIM;//4096;

   /*float*/int8_t *cu_face;
   /*float*/int8_t *cu_database;
   /*float*/int alpha=1; 
   /*float*/int beta=0;
   
  
   /*float*/int8_t *cu_distance;

   cublasHandle_t cu_handle;
   cublasStatus_t  cubStatus = cublasCreate(&cu_handle);
   std::cout << cudaGetErrorEnum(cubStatus) << std::endl;

   CHECK_ERROR(cudaMalloc((void**) &cu_face,feature_dim*face_num*sizeof(/*float*/int8_t)));
   CHECK_ERROR(cudaMalloc((void**) &cu_database ,database_num*feature_dim*sizeof(/*float*/int8_t)));
   CHECK_ERROR(cudaMalloc((void**) &cu_distance,database_num*face_num*sizeof(/*float*/int8_t)));

   int8_t *h_A=(int8_t*)malloc(face_num*feature_dim*sizeof(int8_t));
   int8_t *h_B=(int8_t*)malloc(database_num*feature_dim*sizeof(int8_t));
   int8_t *h_C=(int8_t*)malloc(face_num*database_num*sizeof(int8_t));
   /* Fill the matrices with test data */

    for (int i = 0; i <(face_num*feature_dim); i++)
    {
        h_A[i] = 1;      
    }

    for (int i = 0; i <(database_num*feature_dim); i++)
    {
        h_B[i] = 2;      
    }

    for (int i = 0; i <(face_num*database_num); i++)
    {
        h_C[i] = 0;      
    }
   /* Initialize the device matrices with the host matrices */
    //cublasStatus_t = 
    //cublasSetVector((face_num*feature_dim), sizeof(h_A[0]), h_A, 1, cu_face, 1);
    //cudaMemcy()
    cudaMemcpy(cu_face, h_A, face_num*feature_dim*sizeof(h_A[0]),cudaMemcpyHostToDevice);
    /*if (cublasStatus_t != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }*/

    //cublasStatus_t =
     //cublasSetVector((database_num*feature_dim), sizeof(h_B[0]), h_B, 1, cu_database, 1);
    cudaMemcpy(cu_database, h_B, database_num*feature_dim*sizeof(h_B[0]),cudaMemcpyHostToDevice);
    /*if (cublasStatus_t != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }*/

   // cublasStatus_t =
    //cublasSetVector(face_num*database_num, sizeof(h_C[0]), h_C, 1, cu_distance, 1);

    /*if (cublasStatus_t != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }*/

   //cublasHandle_t cu_handle;
   //cublasStatus_t  cubStatus = cublasCreate(&cu_handle);
   //std::cout << cudaGetErrorEnum(cubStatus) << std::endl;
   
   float gpu_time;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   //cubStatus  = cublasSgemm(cu_handle, CUBLAS_OP_T,CUBLAS_OP_T,face_num,1,feature_dim, &alpha, cu_database, feature_dim, cu_face, 1, &beta, cu_distance, face_num);
   //std::cout << cudaGetErrorEnum(cubStatus) << std::endl;
   for(int t=0;t<1;t++){
   cubStatus = cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    database_num,face_num,feature_dim,//n[i], m[i], k[i],
                    &alpha, cu_database/*d_B*/, 
                    CUDA_R_8I, database_num/*face_num*/,//n[i], 
                    cu_face, CUDA_R_8I, feature_dim,//k[i], 
                    &beta, cu_distance/*d_C*/, 
                    CUDA_R_32I, database_num,//n[i],
                    CUDA_R_32I,   // specify the computatioin type for cublasGemmEx
                    CUBLAS_GEMM_DFALT); // specify the algorithm for cublasGemmEx
                    //CUBLAS_GEMM_ALGO2);           // specify the algorithm for cublasGemmEx
   
   //cudaThreadSynchronize();
   }
   //std::cout << cudaGetErrorEnum(cubStatus) << std::endl;
   //gpu_t=(cutGetTimerValue(timer1)-t0)/1000.0f;
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&gpu_time, start, stop);
   cout <<"GPU Time [s] " << gpu_time << endl;

   //int8_t *h_A=(int8_t*)malloc(face_num*feature_dim*sizeof(int8_t));
   //int8_t *h_B=(int8_t*)malloc(database_num*feature_dim*sizeof(int8_t));
   //int8_t *h_C=(int8_t*)malloc(face_num*database_num*sizeof(int8_t));
   int8_t *h_C_ref=(int8_t*)malloc(face_num*database_num*sizeof(int8_t));

   //t0=cutGetTimerValue(timer1);
   //simple_sgemm(face_num,database_num,feature_dim,h_A, h_B, h_C);
   //cpu_t=(cutGetTimerValue(timer1)-t0)/1000.0f;

   //std::cout << cudaGetErrorEnum(cubStatus) << std::endl;
      
   printf(" GPU=%.6fs(%.3fGflops)\n", gpu_time/1000, 1e-6*face_num*database_num*feature_dim*2/(gpu_time));
   //cudaFree(cu_database); 
   //cudaFree(cu_face);
   //cudaFree(cu_distance)I;

   //cublasDestroy(cu_handle);
   /* Read the result back */
   //cublasGetVector((face_num*database_num), sizeof(h_C_ref[0]), cu_distance, 1, h_C_ref, 1);
   
   cudaMemcpy(h_C_ref, cu_database, database_num*face_num*sizeof(h_C_ref),cudaMemcpyDeviceToHost);

   for(int k=0;k<100;++k)
       printf("%f ",float(h_C_ref[k])); 
   //simple_sgemm(DIM, alpha, h_A, h_B, beta, h_C);

   /* Check result against reference */
 /*float  error_norm = 0;
   float ref_norm = 0;

    for (int i = 0; i <face_num*database_num; ++i)
    {
        float diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);
    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }
   */
   cudaFree(cu_database); 
   cudaFree(cu_face);
   cudaFree(cu_distance);

   cublasDestroy(cu_handle);
   return 0;
}
