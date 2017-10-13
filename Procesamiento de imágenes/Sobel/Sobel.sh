#!/bin/bash

#SBATCH --job-name=Sobel
#SBATCH --output=Sobel.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

imageFolder="../../images"

i=0
for image in $imageFolder/*
do
  # echo "image : ${image}"
  for i in `seq 1 20`;
  do
    ./global.out ${image}
    # echo $i
  done
done
