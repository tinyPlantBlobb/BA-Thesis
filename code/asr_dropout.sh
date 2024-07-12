#!/bin/bash
# running a distributed job with PyTorch
#SBATCH --job-name=whisper-eval
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=whispereval.txt

# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 1:4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
asr_dropout.py 

#srun torchrun --nnodes=1:4 --nproc-per-node=$NUM_TRAINERS --max-restarts=3 --rdzv-id=$JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR
# Before job completes save results on a workspace
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/