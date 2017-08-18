#!/bin/bash
#!/bin/bash
#SBATCH --job-name=mult
#SBATCH --output=mult.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./mult ../C/in1.in ../C/in2.in
