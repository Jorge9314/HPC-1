#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0
//porque en opencv lo trabajan como BGR
using namespace cv;


__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){//mientras no se pase de los tamaños de la imagen
        //imageOutput es la imagen resultante definida con filas y columnas en pixeles
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;//aquí está guardando cada canal de cada pixel convertido en escala de grises
    }
}


int main(int argc, char **argv){//recibe un archivo
    cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;//para medir el tiempo
    double cpu_time_used, gpu_time_used;//para la aceleración del algoritmo (cpu/gpu)

    //The smallest data type possible is char, which means one byte or 8 bits.
    //This may be unsigned (so can store values from 0 to 255) or signed (values from -127 to +127)

    char* imageName = argv[1];//en esta línea del archivo se encuentra el nombre de la imagen
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
    Mat image;//matriz???
    image = imread(imageName, 1);//imread recibe la dirección de la imagen, 

    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();//objeto s de tipo Size

    int width = s.width;//ancho de la imagen leída
    int height = s.height;//alto de la imagen leída
    int size = sizeof(unsigned char)*width*height*image.channels();//canales de una imagen a color: rojo, azul, verde
    int sizeGray = sizeof(unsigned char)*width*height;//imagen resultante en escala de grises, debe tener las mismas dimensiones que la imagen original leída


    dataRawImage = (unsigned char*)malloc(size);//reservamos memoria en el host para dataRawImage
    error = cudaMalloc((void**)&d_dataRawImage,size);//reservamos memoria en el device para 
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }

    h_imageOutput = (unsigned char *)malloc(sizeGray);//reservamos memoria para el host, esta es la imagen que queda en escala de grises
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }


    dataRawImage = image.data;//hace una compia de la imagen que enviamos como argumento a dataRawImage

    /*for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dataRawImage[(i*width+j)*3+BLUE] = 0;
        }
    }*/

    startGPU = clock();
    error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);//copia de dataRawImage a d_dataRawImage
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }

    int blockSize = 32;//hilos por bloque
    dim3 dimBlock(blockSize,blockSize,1);//bloque de 32 x 32 hilos = 1024 hilos
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1); //malla de n x n bloques
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);//lanza el kernel
    cudaDeviceSynchronize();//espera a que se ejecute la función en el device
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);//copia imagen resultante (en escala de grises)
    endGPU = clock();

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);//8-bit unsigned type with one channel
    gray_image.data = h_imageOutput;

    start = clock();
    Mat gray_image_opencv;//imagen escala de grises que va a convertir con opencv
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);//copia image en gray_image_opencv y este lo convierte a escala de grises
    end = clock();


    imwrite("./Gray_Image.jpg",gray_image);//guarda la imagen 

    namedWindow(imageName, WINDOW_NORMAL);//nombra a la ventana de la imagen con el nombre que recibimos como parámetro
    namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);//muestra image(que es la imagen que enviamos como argumento) con el nombre de la imagen
    imshow("Gray Image CUDA", gray_image);//muestra la imagen que se convirtió a escala de grises con CUDA
    imshow("Gray Image OpenCV",gray_image_opencv);//muestra la imagen que se convirtió a escala de grises con openv

    waitKey(0);//espera a que oprimamos alguna letra ( wait for keypress infinitely)

    //free(dataRawImage);
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;//calcula tiempos y aceleramiento
    printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
    printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    cudaFree(d_dataRawImage);//librea memoria
    cudaFree(d_imageOutput);
    return 0;
}

/*
Mat is basically a class with two data parts: the matrix header (containing information such as the size of the matrix, the method used for storing, 
at which address is the matrix stored, and so on) and a pointer to the matrix containing the pixel values (taking any dimensionality depending on the 
method chosen for storing) . The matrix header size is constant, however the size of the matrix itself may vary from image to image and usually is larger 
by orders of magnitude.
http://docs.opencv.org/doc/tutorials/core/mat_the_basic_image_container/mat_the_basic_image_container.html

CV_<bit-depth>{U|S|F}C(<number_of_channels>)

Mat::depth
Returns the depth of a matrix element.

C++: int Mat::depth() const
The method returns the identifier of the matrix element depth (the type of each individual channel). For example, for a 16-bit signed element array, the method returns CV_16S . A complete list of matrix types contains the following values:

CV_8U - 8-bit unsigned integers ( 0..255 )
CV_8S - 8-bit signed integers ( -128..127 )
CV_16U - 16-bit unsigned integers ( 0..65535 )
CV_16S - 16-bit signed integers ( -32768..32767 )
CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )


Mat::channels
Returns the number of matrix channels.

C++: int Mat::channels() const
The method returns the number of matrix channels.


*/
