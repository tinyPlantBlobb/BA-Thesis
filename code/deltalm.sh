#!/bin/bash
#SBATCH --job-name=deltalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=deltaeval.txt

SPMMODEL=$1/spm.model
source qe-whitebox/bin/activate
ws=$2
output=$3
base=$PWD
dlmmodel=$4
#export SPMMODEL
#ws=$(ws_find finals)/
rm $2/data-bin/dropout.spm*
rm $2/data-bin/dropout/test.*
source nmt.sh $1 $2 $3 $4
cd $3
rename generate-test.txt dropout-test.txt generate-test.txt
cd $base

source dropoutlessnmt.sh $1 $2 $3 $4
