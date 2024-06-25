
import torch
import pandas as pd
import urllib
import tarfile
import torchaudio

#get the data ready for the Models 
tar_path = download_asset("dataset/IWSLT23.tst2023.en-de.tar.gz")
tar_item = "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/ted_13587.wav"
with tarfile.open(tar_path, mode="r") as tarfile_:
    fileobj = tarfile_.extractfile(tar_item)
    waveform, sample_rate = torchaudio.load(fileobj)