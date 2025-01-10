#!/bin/bash
#SBATCH --job-name=deltalmdponce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=deltadponceeval.txt
SPMMODEL=$1/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$2
output=$3
base=$PWD
dlmmodel=$4
python spm.py $ws/data-bin/test.de $ws/test.spm.de $SPMMODEL
python spm.py $ws/data-bin/test.en $ws/test.spm.en $SPMMODEL

fairseq-preprocess \
  --source-lang en --target-lang de \
  --bpe sentencepiece \
  --testpref $ws/test.spm \
  --destdir $ws \
  --workers 20 \
  --srcdict $ws/dict.en.txt --tgtdict $ws/dict.de.txt

echo "preprocessing done"

python $base/deltalm/unilm/deltalm/generate.py $ws/ \
  --path $dlmmodel \
  --arch deltalm_large --model-overrides "{'pretrained_deltalm_checkpoint': '$ws/deltalm-large.pt'}" \
  --source-lang en --target-lang de --batch-size 1 --beam 1 \
  --remove-bpe=sentencepiece --unkpen 5 --results-path $output

rm $ws/test.spm.*
#rm $ws/data-bin/data/test*
