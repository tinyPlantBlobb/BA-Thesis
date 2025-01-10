#!/bin/bash
#SBATCH --job-name=endtoendseamless
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=13:30:00
#SBATCH --output=seamlesendtoend.txt

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
mkdir -p $TMPDIR/data
tar -C $TMPDIR/data -vxzf $(ws_find finals)/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate

#pip install transformers --upgrade
#pip install datasets --upgrade
#pip install evaluate --upgrade
#pip install librosa --upgrade
#pip install sentencepiece
#pip install protobuf
cd $TMPDIR
mkdir results
cd 
#cp ~/dropoutfulltranscriptions.csv $TMPDIR/results/dropoutfulltranscriptions.csv
srun torchrun --nnodes 1 --nproc_per_node 1 seamlessendtoend.py
ls $TMPDIR
ls $TMPDIR/results
#rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/

rsync -av $TMPDIR/results $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
