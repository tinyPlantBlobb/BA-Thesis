#!/bin/bash
#SBATCH --job-name=evaluations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=2:30:00
#SBATCH --output=baseeval.txt

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo Head node: $head_node
head_node_ip=$(srun --nodes=1 --time=1:00 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
mkdir -p $TMPDIR/data
tar -C $TMPDIR/data -vxzf $(ws_find iswslt-dataset)/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate

pip install transformers
pip install datasets 
pip install evaluate
pip install librosa
pip install sentencepiece
pip install protobuf
pip install unbabel-comet
pip install jiwer

srun torchrun --nnodes 1 --nproc_per_node 1 asr_regular.py
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python srun torchrun --nnodes 1 --nproc_per_node 1 seamless_regular.py
#srun torchrun --nnodes 1 --nproc_per_node 1 mnt_part.py
cp $TMPDIR/data-bin/* $(ws_find iswslt-dataset)/data-bin/
# 1. param = path to the dir that contains the dataset in the deltalm split format

#PRETRAINEDMODEL=/project/OML/dliu/iwslt2023/model/mt/deltalm-large.tune.bilingual.de.diversify.adapt.TEDonly.clean/checkpoint_avg_last5.pt
#SPMMODEL=$(ws_find iswslt-dataset)/spm.model

#srun python sentencepiece.py $(ws_find iswslt-dataset)/data-bin/test.de $(ws_find iswslt-dataset)/data-bin/test.spm.de
#srun python sentencepiece.py $(ws_find iswslt-dataset)/data-bin/test.en $(ws_find iswslt-dataset)/data-bin/test.spm.en
#
#fairseq-preprocess \
#  --source-lang en --target-lang de \
#  --trainpref data-bin/train --validpref data-bin/valid --testpref data-bin/test \
#  --destdir data-bin/data \
#  --workers 20 \
#  --srcdict spm.model --tgtdict spm.model
#
#srun fairseq-generate $(ws_find iswslt-dataset)/data-bin/data \
#  --path /project/OML/dliu/iwslt2023/model/mt/deltalm-large.tune.bilingual.de.diversify.adapt.TEDonly.clean/checkpoint_avg_last5.pt \
#  --source-lang eng --target-lang deu \
#  --batch-size 128 --beam 5 --remove-bpe --results-path $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID} | tee $TMPDIR/results/dlmtranscriptions.csv
#
srun python evaluations.py

# Before job completes save results on a workspace
#rsync -av $TMPDIR/results/resultscore.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results/scores.txt $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/

rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
rsync -av $TMPDIR/results/ $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
