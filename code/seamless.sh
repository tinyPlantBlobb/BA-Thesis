#!/bin/bash
#SBATCH --job-name=endtoendseamless
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=13:30:00
#SBATCH --output=seamlesendtoend.txt

base=PWD
mkdir -p $TMPDIR/data
tar -C $TMPDIR/data -vxzf $(ws_find finals)/segments_IWSLT-23.en-de.tar.gz
source qe-whitebox/bin/activate

cd $TMPDIR
mkdir results
cd $base
#cp ~/dropoutfulltranscriptions.csv $TMPDIR/results/dropoutfulltranscriptions.csv
srun torchrun --nnodes 1 --nproc_per_node 1 seamlessendtoend.py
#rsync -av $TMPDIR/results/fulltranscriptions.csv $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/

#rsync -av $TMPDIR/results $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
