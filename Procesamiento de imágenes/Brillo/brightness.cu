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

  Size s = image.size();

  int width = s.width, height = s.height;
  int sz = sizeof(unsigned char)*width*height*image.channels();
  int size = sizeof(unsigned char)*width*height; // image with brightness

  h_inputImage = (unsigned char*)malloc(sz);
  h_inputImage = image.data;

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
  h_outputImageCopy = (unsigned char*)malloc(size);

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
/*
  error = cudaMemcpy(h_outputImage, d_outputImage, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
    printf("Error copying d_outputImage to h_outputImageCopy\n");
    exit(-1);
  }
*/
/*
  Mat result;
  result.create(height, width, CV_8UC1);
  result.data = h_outputImage;

  //imshow("Grises",resultado_gray_image);

  imwrite("out.png",result);
*/
/*
  //Imagen escala de grises CPU

  //Separamos memoria para h_ImagenGrises

  h_ImagenGrises = (unsigned char*)malloc(size);

  brightness(h_ImagenInicial, width, height, h_ImagenGrises);

  //Mostramos la imagen en escala de grises de CPU
  Mat resultado_gray_imageCPU;
  resultado_gray_imageCPU.create(height,width,CV_8UC1);
  resultado_gray_imageCPU.data = h_ImagenGrises;

  //imwrite("./outputs/1112786793.png",resultado_gray_imageCPU);
  imshow("Grises CPU",resultado_gray_imageCPU);
  waitKey(0);


  free(h_ImagenInicial);free(h_img_gray);
  cudaFree(d_ImagenInicial);cudaFree(d_img_gray);
  free(h_ImagenGrises);
*/
  return 0;
}
