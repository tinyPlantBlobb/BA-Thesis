#!/bin/bash 
#SBATCH --job-name=segment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --time=3:30:00
#SBATCH --output=segmenter.txt
source ~/qe-whitebox/bin/activate
source  ~/mwerSegmenter/segmentBasedOnMWER.sh ~/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.en.xml  ~/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.de.xml ~/dlmprint.txt dlm German ~/dlmsegmented.xml normalise 1

