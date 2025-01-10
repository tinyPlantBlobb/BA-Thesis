#!/bin/bash
python3 -m venv qe-whitebox
source qe-whitebox/bin/activate
pip install pytorch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install evaluate
pip install librosa
pip install matplotlib
pip install numpy
pip install sentencepiece
pip install tqdm
pip install protobuf
pip install unbabel-comet
pip install jiwer
pip install werpy

mkdir deltalm
cd deltalm/
git clone -n --depth=1 --recursive https://github.com/microsoft/unilm.git
cd unilm/
git sparse-checkout set --no-cone deltalm
git checkout
git submodule update --init deltalm/fairseq
cd deltalm/
curl https://raw.githubusercontent.com/tinyPlantBlobb/fairseq/refs/heads/main/fairseq/sequence_generator.py -o fairseq/fairseq/sequence_generator.py
curl https://raw.githubusercontent.com/tinyPlantBlobb/fairseq/refs/heads/main/fairseq_cli/generate.py -o fairseq/fairseq_cli/generate.py

pip install fairseq/
