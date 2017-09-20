# Multiplicaciones de matrices


## Programas auxiliares

[compareFiles.c](https://github.com/carolinajimenez26/HPC/blob/master/Multiplicacion%20de%20matrices/C/compareFiles.c) es un comparador de archivos, se pueden comparar los archivos
de salida de las multiplicaciones.

### Para  compilarlo

```bash
gcc compareFiles.c -o compare.out
```

### Para ejecutarlo

Se deben enviar por parámetro el nombre de los dos archivos a comparar.

```bash
./a.compare matrix1.in matrix2.in
```

[createMatrixfile.c](https://github.com/carolinajimenez26/HPC/blob/master/Multiplicacion%20de%20matrices/C/createMatrixfile.c) es un generador de matrices. La salida de este programa
será algo como esto:

2
3
1.1,2.4,3.3
4.8,5.2,6.7

Donde los dos primeros números es la cantidad de filas y columnas de la matriz,
y a continuación los datos de las matrices separados por coma (números aleatorios).

### Para  compilarlo

```bash
gcc createMatrixfile.c -o create.out
```

### Para ejecutarlo

Se deben enviar por parámetro el nombre del archivo de salida.

```bash
./create.out matrix1.in matrix2.in
```

Una vez ejecutado debe ingresarse las filas y las columnas deseadas.
