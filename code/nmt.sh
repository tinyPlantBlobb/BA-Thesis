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
# todo fix path for model checkpoint (ask danni)
srun fairseq-generate $(ws_find iswslt-dataset)/data-bin/data/ \
  --path /project/OML/dliu/iwslt2023/model/mt/deltalm-large.tune.bilingual.de.diversify.adapt.TEDonly.clean/checkpoint_avg_last5.pt \
  --source-lang en --target-lang de \
  --batch-size 128 --beam 5 --remove-bpe --results-path $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID} | tee $TMPDIR/results/dlmtranscriptions.csv
