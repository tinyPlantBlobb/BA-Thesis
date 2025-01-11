#!/bin/bash
#SBATCH --job-name=dropouteval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=9:30:00
#SBATCH --output=dropouteval.txt
ws=$2
tar -C $TMPDIR/data -vxzf $ws/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate
output=$3
TMPDIR=$1
input=$TMPDIR/results
curl https://raw.githubusercontent.com/google/sentencepiece/refs/heads/master/python/src/sentencepiece/sentencepiece_model_pb2.py -o qe-whitebox/lib64/python3.9/site-packages/sentencepiece/sentencepiece_model_pb2.py
torchrun --nnodes 1 --nproc_per_node 1 asr_dropout.py
echo "\n dropout done \n"
rsrun torchrun --nnodes 1 --nproc_per_node 1 seamless.py
#echo "seamless done"
#cd $TMPDIR
python dropoutevaluations.py $output

#rsync -av $TMPDIR/results/scores.txt $(ws_find finals)/results-${SLURM_JOB_ID}/
