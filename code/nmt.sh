SPMMODEL=$(ws_find iswslt-dataset)/spm.model
source qe-whitebox/bin/activate
ws=$(ws_find iswslt-dataset)
srun sentencepiece.py data-bin/test.de data-bin/test.spm.de
srun sentencepiece.py data-bin/test.en data-bin/test.spm.en

fairseq-preprocess \
  --source-lang en --target-lang de \
  --trainpref data-bin/train --validpref data-bin/valid --testpref data-bin/test \
  --destdir data-bin/data \
  --workers 20 \
  --srcdict spm.model --tgtdict spm.model

srun fairseq-generate $(ws_find iswslt-dataset)/data-bin/ \
  --path /project/OML/dliu/iwslt2023/model/mt/deltalm-large.tune.bilingual.de.diversify.adapt.TEDonly.clean/checkpoint_avg_last5.pt \
  --source-lang en --target-lang de \
  --batch-size 128 --beam 5 --remove-bpe --results-path. qe-whitebox/bin/activate $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID} | tee $TMPDIR/results/dlmtranscriptions.csv
