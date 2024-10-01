SPMMODEL=$(ws_find iswslt-dataset)/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$(ws_find iswslt-dataset)
srun python spm.py $ws/data-bin/test.de $ws/data-bin/test.spm.de $SPMMODEL
srun python spm.py $ws/data-bin/test.en $ws/data-bin/test.spm.en $SPMMODEL

echo "building done"

fairseq-preprocess \
  --source-lang en --target-lang de \
  --testpref $ws/data-bin/test \
  --destdir $ws/data-bin/data \
  --workers 20 \
  --srcdict $ws/data-bin/dict.txt --tgtdict $ws/data-bin/dict.txt

echo "preprocessing done"

srun python ~/deltalm/generate.py $(ws_find iswslt-dataset)/data-bin/data \
  --path checkpoint_avg_last5.pt \
  --model-overrides "{'pretrained_deltalm_checkpoint': '~/deltalm-large.pt'}" \
  --source-lang en --target-lang de \
  --batch-size 128 --beam 5 --remove-bpe --results-path $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID} | tee $TMPDIR/results/dlmtranscriptions.csv
