#!/bin/bash
#SBATCH --job-name=whisper-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=130:00
#SBATCH --output=whispereval.txt

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
mkdir -p $TMPDIR/data
tar -C $TMPDIR/data -vxzf $(ws_find iswslt-dataset)/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate

pip install transformers --upgrade
pip install datasets --upgrade
pip install evaluate --upgrade
pip install librosa --upgrade
pip install sentencepiece
pip install protobuf
pip install unbabel-comet

srun torchrun --nnodes 1 --nproc_per_node 1 asr_regular.py
srun torchrun --nnodes 1 --nproc_per_node 1 seamless_regular.py
#srun torchrun --nnodes 1 --nproc_per_node 1 mnt_part.py

# 1. param = path to the dir that contains the dataset in the deltalm split format
# TODO output file angeben
#srun fairseq-generate $(ws_find iswslt-dataset)/data-bin/ \
#    --path $(ws_find iswslt-dataset)/checkpoints/deltalm-large.pt \
#    --batch-size 128 --beam 5 --remove-bpe --resluts-path $/ws_find iswslt-dataset)/results-${SLURM_JOB_ID}| tee $TMPDIR/results/fulltranscriptions.csv

# Before job completes save results on a workspace
rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
