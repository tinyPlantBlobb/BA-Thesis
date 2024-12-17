#!/bin/bash
#SBATCH --job-name=seamlesssdropoutevalonce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=0:25:00
#SBATCH --output=seamlessdropoutounceeval.txt

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
ws=$(ws_find finals)
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
mkdir -p $TMPDIR/data
mkdir $TMPDIR/results
source qe-whitebox/bin/activate
cp $ws/results-24973453/dropoutfulltranscriptions.csv $TMPDIR/results/
curl https://raw.githubusercontent.com/google/sentencepiece/refs/heads/master/python/src/sentencepiece/sentencepiece_model_pb2.py -o /pfs/data5/home/kit/stud/utqma/qe-whitebox/lib64/python3.9/site-packages/sentencepiece/sentencepiece_model_pb2.py 
torchrun --nnodes 1 --nproc_per_node 4 dropoutseamless.py
echo "seamless done"
cd $TMPDIR
#srun python ~/dropoutevaluations.py
#rsync -av $TMPDIR/results/scores.txt $(ws_find finals)/results-${SLURM_JOB_ID}/

rsync -av $TMPDIR/results/dropoutfulltranscriptions.csv $(ws_find finals)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results/ $(ws_find finals)/results-${SLURM_JOB_ID}/
