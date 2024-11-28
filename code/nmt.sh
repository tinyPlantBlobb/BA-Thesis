#!/bin/bash
#SBATCH --job-name=deltalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=deltaeval.txt

SPMMODEL=$(ws_find iswslt-dataset)/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$(ws_find iswslt-dataset)
srun python spm.py $ws/data-bin/test.de $ws/data-bin/test.spm.tgt $SPMMODEL
srun python spm.py $ws/data-bin/test.en $ws/data-bin/test.spm.src $SPMMODEL

echo "building done"

fairseq-preprocess \
  --source-lang en --target-lang de \
  --testpref $ws/data-bin/test \
  --destdir $ws/data-bin \
  --workers 20 \
  --srcdict $ws/dict.txt --tgtdict $ws/dict.txt

echo "preprocessing done"

srun python ~/deltalm/generate.py $(ws_find iswslt-dataset)/data-bin \
  --path ~/checkpoint_avg_last5.pt \
  --arch deltalm_large --model-overrides "{'pretrained_deltalm_checkpoint': 'deltalm-large.pt'}" \
  --source-lang en --target-lang de --batch-size 128 --beam 5\
  --remove-bpe=sentencepiece --unkpen 5 --results-path $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID} --retain-dropout --retain-dropout-modules '["TransformerModel", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer"]' 
echo "run one done"

