

PRETRAINED_MODEL=/home/plantpalfynn/uni/BA/BA-Thesis/code/finetuneDLM/model.pt
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