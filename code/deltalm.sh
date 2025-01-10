#!/bin/bash
#SBATCH --job-name=deltalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=deltaeval.txt

SPMMODEL=$(ws_find finals)/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
#ws=$(ws_find finals)/
rm $(ws_find finals)/data-bin/dropout.spm*
rm $(ws_find finals)/data-bin/dropout/test.*
source nmt.sh
cd $(ws_find finals)/results-${SLURM_JOB_ID}/
rename generate-test.txt dropout-test.txt generate-test.txt
cd 
source dropoutlessnmt.sh
