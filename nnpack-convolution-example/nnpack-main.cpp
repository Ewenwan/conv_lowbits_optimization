#include <iostream>  
#include "nnpack.h"  
#include <ctime>  
#include <vector>  
#include <algorithm> 
//using namespace std;  
#include <iomanip>

#include <iostream>
#include <chrono>
#include <ctime>
using namespace std;

//using namespace cv;


template <typename F>
double benchmark(const F &fun, int n = 1)
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
			    return timings[timings.size() / 2] / 1000;
}

float test_nnpack(){  
    //init nnpack  
    enum nnp_status init_status = nnp_initialize();  
    if (init_status != nnp_status_success) {  
        return 0;  
    }  
  
    enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_ft8x8;//ft8x8;//nnp_convolution_algorithm_auto;  
    enum nnp_convolution_transform_strategy strategy=nnp_convolution_transform_strategy_tuple_based;  
    const size_t batch_size = 1;  
    const size_t input_channels = 32;  
    const size_t output_channels = 64;  
    const struct nnp_padding input_padding = { 1, 1, 1, 1 };  
    const struct nnp_size input_size ={ 256, 256};  
    const struct nnp_size kernel_size = { 5, 5 };  
    const struct  nnp_size stride={.width=2,.height=2};  
    const struct nnp_size output_size = {  
            .width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width)/stride.width + 1,  
            .height =(input_padding.top + input_size.height + input_padding.bottom - kernel_size.height)/stride.height  + 1  
    };  
  
  
    //malloc memory for input, kernel, output, bias  
    float* input = (float*)malloc(batch_size * input_channels *input_size.height *input_size.width * sizeof(float));  
    float* kernel = (float*)malloc(input_channels * output_channels * kernel_size.height * kernel_size.width * sizeof(float));  
    float* output = (float*)malloc(batch_size* output_channels * output_size.height * output_size.width * sizeof(float));  
    float* bias = (float*)malloc(output_channels * sizeof(float));  
  
    pthreadpool_t threadpool= nullptr;  
  
  
    struct nnp_profile computation_profile;//use for compute time;  
    //init input data  
    int i,j;  
    for(int c=0; c<input_channels;c++ ){  
        for(i=0; i<input_size.height; i++){  
            for(j=0; j<input_size.width; j++){  
                input[c*input_size.height*input_size.width+i*input_size.width+j] = (i*input_size.width+j)*0.1;  
            }  
        }  
    }  
  
    //init kernel data  
    for(int i=0; i<output_channels;i++ ){  
        for(j=0; j<input_channels*kernel_size.height*kernel_size.width; j++){  
            kernel[i*input_channels*kernel_size.height*kernel_size.width+j] = 0.1;  
        }  
    }  
  
    //init bias data  
    for(int i=0; i<output_channels;i++ ){  
        bias[i] = 1.0;  
    }  
  
    //execute conv  
    std::cout << "nnpact convolution:       " << std::setw(10) << benchmark([&]() {  
        nnp_convolution_inference(algorithm,  
                                  strategy,  
                                  input_channels,  
                                  output_channels,  
                                  input_size,  
                                  input_padding,  
                                  kernel_size,  
                                  stride,  
                                  input,  
                                  kernel,  
                                  bias,  
                                  output,  
                                  threadpool,  
                                  nullptr);  
    
    }) << " ms" << std::endl;
    
    std::vector<float>out;  
    for(int i=0;i<output_channels*output_size.height*output_size.width;i++){  
        out.push_back(output[i]);  
    }  
  
    return 1;//conv_time_use;  
}  
int main() {  
    cout << test_nnpack()<< endl;  
    return 0;  
}  
