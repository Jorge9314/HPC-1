#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<opencv2/opencv.hpp>
#include<cuda.h>
using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

__host__
void brightness(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
  for(int row = 0; row < height; row++){
    for(int col = 0; col < width; col++){
      imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*2 + imageInput[(row*width+col)*3+GREEN]*2 + imageInput[(row*width+col)*3+BLUE]*2;
    }
  }
}

__global__
void brightnessCU(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  if((row < height) && (col < width)) {
    imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*2 + imageInput[(row*width+col)*3+GREEN]*2 + imageInput[(row*width+col)*3+BLUE]*2;
  }

}

int main(int argc, char **argv){

  cudaError_t error = cudaSuccess;
  unsigned char *h_inputImage, *d_inputImage, *h_outputImage, *d_outputImage, *h_outputImageCopy;
  char* imageName = argv[1];

  if (argc !=2) {
    printf("Path of the image must be specified!!\n");
    return 1;
  }

  Mat image = imread(imageName, 1);

  if (!image.data) {
    printf("No image Data\n");
    return 1;
  }

  // imshow("Image",image);

  Size s = image.size();

  int width = s.width, height = s.height;
  int sz = sizeof(unsigned char)*width*height*image.channels();
  int size = sizeof(unsigned char)*width*height; // image with brightness

  h_inputImage = (unsigned char*)malloc(sz);
  h_inputImage = image.data;

  // ---------------------- GPU --------------------------------

  error = cudaMalloc((void**)&d_inputImage,sz);
  if(error != cudaSuccess){
    cout << "Error allocating memory for d_inputImage" << endl;
    exit(-1);
  }

  error = cudaMemcpy(d_inputImage, h_inputImage, sz, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    printf("Error copying h_inputImage to d_inputImage\n");
    exit(-1);
  }

  h_outputImage = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_outputImage, size);
  if(error != cudaSuccess){
    printf("Error allocating memory for d_outputImage\n");
    exit(-1);
  }

  int blockSize = 32.0;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/blockSize), ceil(height/blockSize), 1);
  brightnessCU<<<dimGrid,dimBlock>>>(d_inputImage, width, height, d_outputImage);
  cudaDeviceSynchronize();

  error = cudaMemcpy(h_outputImage, d_outputImage, size, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
    printf("Error copying d_outputImage to h_outputImageCopy\n");
    exit(-1);
  }


  Mat resultGPU;
  resultGPU.create(height, width, CV_8UC1);
  resultGPU.data = h_outputImage;

  //imshow("Grises",resultado_gray_image);

  imwrite("outGPU.png",resultGPU);


  // ---------------------- CPU --------------------------
  
  h_outputImageCopy = (unsigned char*)malloc(size);
  brightness(h_inputImage, width, height, h_outputImageCopy);

  Mat resultCPU;
  resultCPU.create(height,width,CV_8UC1);
  resultCPU.data = h_outputImageCopy;

  imwrite("outCPU.png",resultCPU);
  // imshow("CPU",resultado_gray_imageCPU);
  // waitKey(0);


  free(h_inputImage);free(h_outputImage);free(h_outputImageCopy);
  cudaFree(d_inputImage);cudaFree(d_outputImage);
  
  return 0;
}
