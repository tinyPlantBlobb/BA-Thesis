bash examples/prepare_iwslt14.sh /tmp/iwslt14

bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.tokenized.de-en \
     /tmp/iwslt14/iwslt14.spm \
     /path/to/checkpoint/spm.model

bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.spm \
     /tmp/iwslt14/iwslt14.bin \
     /path/to/checkpoint/dict.txt

bash examples/train_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints \
     /path/to/checkpoint/model.pt

bash examples/evaluate_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints