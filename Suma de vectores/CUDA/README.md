# Suma de matrices con CUDA

## Para compilar

```bash
nvcc file_name.cu -o executable_name
```

## Para ejecutar

__Con Slurm:__

```bash
sbatch vecAdd.sh
```

Y el resultado quedará en _vecAdd.out_

__Sin Slurm:__

Se deben enviar por parámetros los archivos de los vectores

```bash
./executable_name vec1.in vec2.in
```
