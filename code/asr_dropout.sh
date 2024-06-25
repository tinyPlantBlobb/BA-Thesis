#!/bin/bash
# very simple example on how to use local $TMPDIR
#SBATCH -N 1
#SBATCH -t 24:00:00


# Extract compressed input dataset on local SSD
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/dataset.tgz

# The application reads data from dataset on $TMPDIR and writes results to $TMPDIR
myapp -input $TMPDIR/dataset/myinput.csv -outputdir $TMPDIR/results

python3 asr_dropout.py --data_dir $TMPDIR/dataset --output_dir $TMPDIR/results

# Before job completes save results on a workspace
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/