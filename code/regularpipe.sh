#!/bin/bash
#SBATCH --job-name=whisper-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=70:00
#SBATCH --output=whispereval.txt

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip 
export LOGLEVEL=INFO
mkdir -p $TMPDIR/data
tar -C $TMPDIR/data -vxzf $(ws_find iswslt-dataset)/segments_IWSLT-23.en-de.tar.gz 
source qe-whitebox/bin/activate

pip install transformers
pip install datasets
pip install evaluate
pip install librosa

srun torchrun --nnodes 1 --nproc_per_node 1 asr_regular.py 
#srun torchrun --nnodes 1 --nproc_per_node 1 seamless.py
#srun torchrun --nnodes 1 --nproc_per_node 1 mnt_part.py

# Before job completes save results on a workspace
rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
tar -cvzf $TMPDIR/results-${SLURM_JOB_ID}.tar.gz $TMPDIR/results
rsync -av $TMPDIR/results-${SLURM_JOB_ID}.tar.gz $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/