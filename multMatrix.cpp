#include<iostream>
#include<malloc.h>
using namespace std;

void multiplicaMatrices(int* X,int filX,int colX,int* Y,int filY,int colY,int* Z){
	for(int i=0;i<filX;i++){
		for(int j=0;j<colY;j++){
			int suma=0;
			for(int k=0;k<filY;k++){
				suma=suma+X[(i*colX)+k]*Y[(k*colY)+j];

			}
			Z[(i*colY)+j]=suma;
		}	
	}
}

void imprime(int* A,int filas, int columnas){//imprime como si fuera una matriz
	for(int i = 0; i < filas; i++){
        	for(int j = 0; j < columnas; j++){
            		cout<<A[(i*columnas)+j]<<" ";
        	}
        cout<<endl;
    }
}	

void inicializa(int *A,int filas, int columnas){//inicializa arreglos
	for(int i=0;i<filas*columnas;i++){
		A[i]=1;
	}
}

int main(void){
	int *A,*B,*C;
	int filA=5,colA=4,filB=4,colB=1;
	A=(int*)malloc(filA*colA*sizeof(int)); 
	B=(int*)malloc(filB*colB*sizeof(int));
	C=(int*)malloc(filA*colB*sizeof(int));

	inicializa(A,filA,colA);
	inicializa(B,filB,colB);
	
	if(colA==filB){//para que sean multiplicables
		multiplicaMatrices(A,filA,colA,B,filB,colB,C);
		imprime(C,filA,colB);
	}else{
		cout<<"Error, no se pueden multiplicar"<<endl;
		return -1;//error en la ejecución del programa
	}
	
	free(A);free(B);free(C);
	return 0;
}
