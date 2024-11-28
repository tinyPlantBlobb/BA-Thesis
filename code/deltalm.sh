#!/bin/bash
#SBATCH --job-name=deltalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --output=deltaeval.txt

SPMMODEL=$(ws_find iswslt-dataset)/spm.model
source qe-whitebox/bin/activate
#export SPMMODEL
ws=$(ws_find iswslt-dataset)
for i in $(seq 0 30);
do 
	source nmt.sh
	cd $(ws_find iswslt-dataset)/results-${SLURM_JOB_ID}/
	rename generate-test.txt generate-test$i.txt generate-test.txt
	cd ~
done
source dropoutlessnmt.sh
