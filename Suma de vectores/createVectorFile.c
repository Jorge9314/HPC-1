#include <stdio.h>
#include <stdlib.h>

typedef char* string;

void fill(float *M, int size) {
  float a = 5.0;
  for (int i = 0; i < size; i++) M[i] = (float)rand() / (float)(RAND_MAX / a);
}

void print(float *M, int size) {
  for (int i = 0; i < size; i++) printf("%.2f ", M[i]);
  printf("\n");
}

void write(float *M, int size, string file_name) {
  FILE *f = fopen(file_name, "w");
  fprintf(f, "%d\n", size);
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  int i, j;
  for (i = 0; i < size; i++) {
    if (i + 1 == size) {
      fprintf(f, "%.2f", M[i]);
    } else {
      fprintf(f, "%.2f,", M[i]);
    }
  }
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc =! 2) {
    printf("Must be called with the name of the out file\n");
    return 1;
  }
  int size;
  string file_name = argv[1];
  printf("File name: %s\n", file_name);
  scanf("%d", &size);
  float *M = (float*)malloc(size*sizeof(float));
  fill(M, size);
  // print(M, size);
  write(M, size, file_name);
  return 0;
}
