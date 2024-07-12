#!/bin/bash
python3 -m venv qe-whitebox
source env/bin/activate
pip install pytorch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers
pip3 install datasets
pip3 install librosa
