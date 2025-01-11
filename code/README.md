# Setup

you will need to have git installed.

The used python version is 3.9

In order to run the code, first run the env_installation.sh script to install the necessary packages. this created a virtual environment and installs the necessary packages. as well as clone and install deltalm and fairseq, specifically a custom version of fairseq that is used for the experiments.

download the spm.model from <https://deltalm.blob.core.windows.net/deltalm/spm.model>
download the deltalm large model from <https://deltalm.blob.core.windows.net/deltalm/deltalm-large.pt>
place both in the workspace directory
the workspace directory is the directory where the data-bin and dictionaries are placed in

to prepare the dataset for running the experiments run the following command with the path to the unzipped IWSLT-23.en-de.zip file
to segemtn the data a yaml file is used, this can be generated with the getyaml funktion in the dataset.py file but it is also included in the repository, if that is used the yaml file should be placed in the IWSLT-23.en-de folder under IWSLT-23.tst2023.en-de/benchmark/en-de/tst2023/

HOME=/path/to/IWSLT-23.en-de.zip python dataset.py

# Seamless based runs

to run the regularpipe.sh script for running whisper and seamless without using dropout, this also creates the base file for running deltalm in data-bin folder.

source regularpipe.sh path/to/IWSLT-23.en-de.zip path/to/workspace path/to/output path/to/deltadatadir

# Dropout experiments

for the dropout experiments, run the asr_dropout.sh script. this will run the whisper and seamless with dropout.
source asr_dropout.sh path/to/segemented_IWSLT-23.en-ed.zip path/to/workspace path/to/output/

# End to end experiments

for the end to end experiments, run the seamless.sh script. this will run seamless in an end to end fashion and generate the results for both the non dropout and dropout experiments.

# Deltalm

before running the deltalm script please run the regularpipe.sh script to generate the transcriptions that are input into deltalm.

to run the deltalm experiments, run the deltalm.sh script.

source deltalm.sh path/to/sentencepiece.model path/to/workspace path/to/output path/to/deltalm-model

make sure the scripts nmt.sh and dropoutlessnmt.sh are in the same directory as the deltalm.sh script
the worspace being the folder where the data-bin, dictionaries and deltalm-large model is placed in
this will run the deltalm model on the generated base file and generate the results for both the non dropout and dropout experiments.

# Evaluation

To evaluate the results, run the evaluation.sh script

source evaluation.sh path/to/seamlessresults path/to/seamless_dropout_results path/to/deltaLm_results

some of the scores and correlation scores are calculated in the plotting.py which generates the plots for the thesis. for this the evaluation.sh script needs to be run first.
to run this run the following command
RES_DIR=path/to/base python plotting.py --seamless path/to/seamlessallscores.txt --dlm path/to/dlmallscores.txt --whisper path/to/fulltransciption.csv --split true
