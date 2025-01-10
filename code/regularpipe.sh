#!/bin/bash
#SBATCH --job-name=evaluations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=5:30:00
#SBATCH --output=baseeval.txt

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

#pip install transformers
#pip install datasets 
#pip install evaluate
#pip install librosa
#pip install sentencepiece
#pip install protobuf
#pip install unbabel-comet
#pip install jiwer
curl https://raw.githubusercontent.com/google/sentencepiece/refs/heads/master/python/src/sentencepiece/sentencepiece_model_pb2.py -o /pfs/data5/home/kit/stud/utqma/qe-whitebox/lib64/python3.9/site-packages/sentencepiece/sentencepiece_model_pb2.py 
srun torchrun --nnodes 1 --nproc_per_node 1 asr_regular.py
srun torchrun --nnodes 1 --nproc_per_node 1 seamless_regular.py
srun python evaluations.py

# Before job completes save results on a workspace
#rsync -av $TMPDIR/results/resultscore.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
#rsync -av $TMPDIR/results/scores.txt $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/

#rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results/ $(ws_find finals)/results-${SLURM_JOB_ID}/
