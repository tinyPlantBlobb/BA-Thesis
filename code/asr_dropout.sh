#!/bin/bash
# running a distributed job with PyTorch
#SBATCH --job-name=whisper-eval
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00
#SBATCH --output=whispereval.txt

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

tar -C $TMPDIR/ -xvzf $(ws_find iswslt-dataset)/dataset.tgz

source qe-whitebox/bin/activate

pip install transformers
pip install datasets
pip install librosa

torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 asr_dropout.py 

# Before job completes save results on a workspace
rsync -av $TMPDIR/results $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/