#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

__host__
void brightness(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
  for(int row = 0; row < height; row++){
    for(int col = 0; col < width; col++){
      imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*2 + imageInput[(row*width+col)*3+GREEN]*2 + imageInput[(row*width+col)*3+BLUE]*2;
    }
  }
}

__global__
void brightness(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  if((row < height) && (col < width)) {
    imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*2 + imageInput[(row*width+col)*3+GREEN]*2 + imageInput[(row*width+col)*3+BLUE]*2;
  }
}

int main(int argc, char **argv){

  cudaError_t error = cudaSuccess;
  unsigned char *h_ImagenInicial, *d_ImagenInicial;
  unsigned char *h_img_gray, *d_img_gray;
  unsigned char *h_ImagenGrises;
  char* imageName = argv[1];
  Mat image;

  image = imread(imageName, 1);

  if(argc !=2 || !image.data) {
    printf("No image Data \n");
    return 1;
  }

  Size s = image.size();

  int width = s.width, height = s.height;
  int sz = sizeof(unsigned char)*width*height*image.channels();
  int size = sizeof(unsigned char)*width*height;//para la imagen en escala de grises

  h_ImagenInicial = (unsigned char*)malloc(sz);

  error = cudaMalloc((void**)&d_ImagenInicial,sz);
  if(error != cudaSuccess){
    cout << "Error reservando memoria para d_ImagenInicial" << endl;
    exit(-1);
  }

  h_ImagenInicial = image.data;

  //Copiamos los datos al device
  error = cudaMemcpy(d_ImagenInicial, h_ImagenInicial, sz, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    printf("Error copiando los datos de h_ImagenInicial a d_ImagenInicial \n");
    exit(-1);
  }

  h_img_gray = (unsigned char*)malloc(size);

  error = cudaMalloc((void**)&d_img_gray,size);
  if(error != cudaSuccess){
    printf("Error reservando memoria para d_img_gray\n");
    exit(-1);
  }

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
  brightness<<<dimGrid,dimBlock>>>(d_ImagenInicial, width, height, d_img_gray);
  cudaDeviceSynchronize();

  //Copiamos datos de la imágen a escala de grises del device al host
  error = cudaMemcpy(h_img_gray,d_img_gray,size, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
    printf("Error copiando los datos de d_img_gray a h_img_gray \n");
    exit(-1);
  }

  //Mostramos la imagen en escala de grises de GPU
  Mat resultado_gray_image;
  resultado_gray_image.create(height,width,CV_8UC1);
  resultado_gray_image.data = h_img_gray;

  //imshow("Grises",resultado_gray_image);

  //imwrite("./outputs/1112786793.png",resultado_gray_image);


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

  return 0;
}
