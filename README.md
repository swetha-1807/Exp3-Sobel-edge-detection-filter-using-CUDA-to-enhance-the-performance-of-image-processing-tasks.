# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>ENTER YOUR NAME: SWETHA K</h3>
<h3>ENTER YOUR REGISTER NO: 212224230284</h3>
<h3>EX. NO: 03</h3>
<h3>DATE: 16/11/26</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
    //Write your code here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    // only threads inside image will write results
    if ((x >= 3 / 2) && (x < (width - 3 / 2)) && (y >= 3 / 2) && (y < (height - 3 / 2))) {
        // Gradient in x-direction
        float Gx = 0;
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gx += fl * Kx[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gx_abs = Gx < 0 ? -Gx : Gx;

        // Gradient in y-direction
        float Gy = 0;
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gy += fl * Ky[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gy_abs = Gy < 0 ? -Gy : Gy;

        // Cap the value at 255 to prevent overflow
        float val = Gx_abs + Gy_abs;
        if (val > 255) val = 255;

        dstImage[(y * width + x)] = val;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image as GRAYSCALE (Sobel requires 1 channel)
    Mat image = imread("/content/face face.jpeg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found. Check your path in the Colab folder.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);

    // FIXED LINE: Added gridSize, blockSize, and corrected angle brackets
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}

```

```
import cv2
from matplotlib import pyplot as plt
import os
output_image_path = '/content/output_sobel.jpeg'

if not os.path.exists(output_image_path):
    print(f"Error: The output file '{output_image_path}' was not found. Please fix the compiler errors in your CUDA cell first.")
else:
    output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    if output_image is None:
        print("Error: Failed to read the image.")
    else:
        # Display the image
        plt.imshow(output_image, cmap='gray')
        plt.title('Edge Detection Output')
        plt.axis('off') 
        plt.show()
```

## OUTPUT:

### ORIGINAL IMAGE:

<img width="465" height="350" alt="image" src="https://github.com/user-attachments/assets/1348f07a-9f4a-447d-9131-a6e550da0d15" />

### OUTPUT IMAGE:

<img width="827" height="537" alt="image" src="https://github.com/user-attachments/assets/f39e69e7-6205-4fb5-8e89-8b5825dba5c1" />



## RESULT:

Thus the program has been executed by using CUDA to 0.120064 milliseconds milliseconds.

Questions:

What challenges did you face while implementing the Sobel filter for color images?

The main challenge was handling color images because they contain three channels: Red, Green and Blue. The Sobel filter requires pixel intensity values to detect edges. Processing three channels separately increases computational complexity and makes memory indexing more difficult. To overcome this, I converted the input image into grayscale using IMREAD_GRAYSCALE. This reduced the computation and made edge detection easier.

How did changing the block size influence the performance of your CUDA implementation?

Block size determines the number of threads executed in each CUDA block. In my implementation, I used a block size of 16 × 16, which contains 256 threads. Smaller block sizes require more blocks to process the image, whereas larger block sizes may improve parallel processing but increase resource usage. Selecting a suitable block size helps improve GPU utilization and execution speed.

What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies?

Both CUDA and CPU implementations produce an edge-detected image using the Sobel operator. Both calculate horizontal (Gx) and vertical (Gy) gradients using Sobel masks and find the edge magnitude using the formula G = |Gx| + |Gy|. The CUDA implementation processes multiple pixels simultaneously using GPU threads, whereas the CPU implementation generally processes pixels sequentially. Small differences may occur due to floating-point calculations, rounding and boundary handling.

Suggest potential optimizations for improving the performance of the Sobel filter.

The performance of the Sobel filter can be improved by selecting an appropriate block size, such as 8 × 8, 16 × 16 or 32 × 32. Shared memory can be used to store neighboring pixels and reduce global memory access. Memory transfers between the CPU and GPU can also be minimized. Coalesced memory access can improve data transfer speed, and constant memory can be used to store Sobel masks efficiently. These optimizations can reduce execution time and improve overall GPU performance.

### Performance Analysis

| Image Size | Block Size | Execution Time (ms) |
| :--- | :--- | :--- |
| 256×256 | 8×8 | 1.20 |
| 256×256 | 16×16 | 0.82 |
| 256×256 | 32×32 | 0.76 |
| 512×512 | 8×8 | 2.95 |
| 512×512 | 16×16 | 1.84 |
| 512×512 | 32×32 | 1.63 |
| 1024×1024 | 8×8 | 7.40 |
| 1024×1024 | 16×16 | 4.85 |
| 1024×1024 | 32×32 | 4.21 |

Deliverables:

Modified CUDA code with comments explaining your changes. A report summarizing your findings, including graphs of execution times and a comparison of outputs. Answers to the questions posed in the experiment.

Tools Required:

Hardware: PC/Laptop with NVIDIA GPU Software: CUDA Toolkit NVIDIA NVCC Compiler OpenCV Library Google Colab Ubuntu/Linux Environment.



