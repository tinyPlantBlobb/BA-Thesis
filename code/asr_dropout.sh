#!/bin/bash
#SBATCH --job-name=dropouteval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:8
#SBATCH --time=8:30:00
#SBATCH --output=dropouteval.txt

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

#pip install transformers 
#pip install datasets 
#pip install evaluate 
#pip install librosa
#pip install sentencepiece
#pip install protobuf
#pip install unbabel-comet
#pip install jiwer
#
srun torchrun --nnodes 1 --nproc_per_node 1 asr_dropout.py
echo "\n dropout done \n"

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python srun torchrun --nnodes 1 --nproc_per_node 1 seamless.py
echo "seamless done"
cd $TMPDIR
#srun python ~/dropoutevaluations.py
rsync -av $TMPDIR/results/scores.txt $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/

rsync -av $TMPDIR/results/dropoutfulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results/ $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
