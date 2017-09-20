# Multiplicacion de matrices con CUDA

```bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Para compilar cualquiera

```bash
nvcc file_name.cu -o executable_name
```

### Memoria global

[global.cu](https://github.com/carolinajimenez26/HPC/blob/master/Multiplicacion%20de%20matrices/CUDA/global.cu) realiza la multiplicación de matrices en la CPU y GPU (en la memoria
global) y compara los resultados de ambas(si son iguales), y si lo son muestra
el tiempo de aceleración generado.

Para la ejecución, se deben enviar por parámetros los nombres de las matrices a
multiplicar

```bash
./a.out matrix1.in matrix2.in
```
