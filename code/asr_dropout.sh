#!/bin/bash
# running a distributed job with PyTorch
#SBATCH --job-name=whisper-eval
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=whispereval.txt

# Extract compressed input dataset on local SSD
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/dataset.tgz

# The application reads data from dataset on $TMPDIR and writes results to $TMPDIR
#myapp -input $TMPDIR/dataset/myinput.csv -outputdir $TMPDIR/results

srun torchrun --nnodes=1:4 --nproc-per-node=$NUM_TRAINERS --max-restarts=3 --rdzv-id=$JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR asr_dropout.py 

# Before job completes save results on a workspace
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/