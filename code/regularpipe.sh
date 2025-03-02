#!/bin/bash
#SBATCH --job-name=evaluations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=5:30:00
#SBATCH --output=baseeval.txt

ws=$1

mkdir -p $ws/data
tar -C $ws/data -vxzf $ws/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate

curl https://raw.githubusercontent.com/google/sentencepiece/refs/heads/master/python/src/sentencepiece/sentencepiece_model_pb2.py -o qe-whitebox/lib64/python3.9/site-packages/sentencepiece/sentencepiece_model_pb2.py
TMPDIR=$1 dlmresults=$4 output=$3 torchrun --nnodes 1 --nproc_per_node 1 asr_regular.py
TMPDIR=$3 output=$3 torchrun --nnodes 1 --nproc_per_node 1 seamless_regular.py
TMPDIR=$3 python evaluations.py
