#!/bin/bash
#SBATCH --job-name=results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=0:20:00
#SBATCH --output=getresults.txt

module load devel/cuda/11.8
nodes=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
source qe-whitebox/bin/activate

#srun python ~/evaluations.py
srun python ~/dlmeval.py
srun python ~/dropoutevaluations.py
