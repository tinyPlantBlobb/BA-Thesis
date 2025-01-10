#!/bin/bash
#SBATCH --job-name=deltalmdponce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=deltadponceeval.txt
SPMMODEL=$(ws_find finals)/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$(ws_find finals) 
srun python spm.py $ws/data-bin/test.de $TMPDIR/test.spm.de $SPMMODEL
srun python spm.py $ws/data-bin/test.en $TMPDIR/test.spm.en $SPMMODEL

fairseq-preprocess \
  --source-lang en --target-lang de \
  --bpe sentencepiece \
  --testpref $TMPDIR/test.spm \
  --destdir $TMPDIR \
  --workers 20 \
  --srcdict $ws/dict.en.txt --tgtdict $ws/dict.de.txt

echo "preprocessing done"

srun python ~/deltalm/generate.py $TMPDIR/ \
  --path ~/checkpoint_avg_last5.pt \
  --arch deltalm_large --model-overrides "{'pretrained_deltalm_checkpoint': 'deltalm-large.pt'}" \
  --source-lang en --target-lang de --batch-size 1 --beam 1 \
  --remove-bpe=sentencepiece --unkpen 5 --results-path $(ws_find finals)/results-${SLURM_JOB_ID}

#rm $ws/data-bin/test.spm.*
#rm $ws/data-bin/data/test*
