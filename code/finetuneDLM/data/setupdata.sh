PRETRAINED_MODEL = /path/to/checkpoint/model.pt
data_bin=/path/to/data-bin/
spm_model=/path/to/checkpoint/spm.model
save_dir= .
batch_size= 4096


bash examples/prepare_iwslt14.sh /tmp/iwslt14


spm_encode --model=$spm_model --output_forexamplesexamplesmat=piece < train.src > train.spm.src
spm_encode --model=$spm_model --output_format=piece < train.tgt > train.spm.tgt
spm_encode --model=$spm_model --output_format=piece < valid.src > valid.spm.src
spm_encode --model=$spm_model --output_format=piece < valid.tgt > valid.spm.tgt
spm_encode --model=$spm_model --output_format=piece < test.src > test.spm.src
spm_encode --model=$spm_model --output_format=piece < test.tgt > test.spm.tgt


bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.tokenized.de-en \
     /tmp/iwslt14/iwslt14.spm \
     /path/to/checkpoint/spm.model


python preprocess.py  \
    --trainpref train.spm \
    --validpref valid.spm \
    --testpref test.spm \
    --source-lang eng --target-lang de \
    --destdir $data_bin \
    --srcdict /path/to/checkpoint/dict.txt \
    --tgtdict /path/to/checkpoint/dict.txt \
    --workers 40


python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_base \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $lr \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 4000 \
    --max-update 400000 \
    --max-epoch 100 \
    --max-tokens $batch_size \
    --update-freq 1 \
    --seed 1 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test

