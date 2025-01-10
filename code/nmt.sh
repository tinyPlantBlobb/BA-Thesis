#!/bin/bash
#SBATCH --job-name=deltalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=12:30:00
#SBATCH --output=deltaeval.txt

SPMMODEL=$1/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$2
output=$3
srun python spm.py $ws/data-bin/dropouttest.de $ws/data-bin/dropout.spm.de $SPMMODEL
srun python spm.py $ws/data-bin/dropouttest.en $ws/data-bin/dropout.spm.en $SPMMODEL

echo "building done"

fairseq-preprocess \
  --source-lang en --target-lang de \
  --bpe sentencepiece \
  --testpref $ws/data-bin/dropout.spm \
  --destdir $ws/data-bin/dropout/ \
  --workers 20 \
  --srcdict $ws/dict.en.txt --tgtdict $ws/dict.de.txt

echo "preprocessing done"

srun python ~/deltalm/generate.py $(ws_find finals)/data-bin/dropout/ \
  --path ~/checkpoint_avg_last5.pt \
  --arch deltalm_large --model-overrides "{'pretrained_deltalm_checkpoint': 'deltalm-large.pt'}" \
  --source-lang en --target-lang de --batch-size 1 --beam 1 --remove-bpe=sentencepiece --unkpen 5 --retain-dropout --results-path $output
#sed -i "/^V-.*$/d" $ws/results-${SLURM_JOB_ID}/generate-test.txt
sed -i "/202\d-\d+.*/d" $output/generate-test.txt
sort -n -t'	' -k 1.3,1 -o generate-test.txt generate-test.txt
echo "run one done"
