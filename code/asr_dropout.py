import numpy as np
import torch.utils
import torch.utils.data

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import urllib
import tarfile
import whisper
import torchaudio

from scipy.io import wavfile
from tqdm import tqdm


#import ipywidgets as widgets
# Whisper
asr_model = whisper.load_model("medium")
options = dict(language="German", beam_size=5, best_of=5, dropout=0.3)
transcribe_options = dict(task="transcribe", **options)
#translate_options = dict(task="translate", **options)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# tar_path = download_asset("dataset/IWSLT23.tst2023.en-de.tar.gz")
# tar_item = "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/ted_13587.wav"
# with tarfile.open(tar_path, mode="r") as tarfile_:
#     fileobj = tarfile_.extractfile(tar_item)
#     waveform, sample_rate = torchaudio.load(fileobj)

def transcriptionProbability(tensor, **options):
    result = {"translationProb": torch.nn.functional.softmax(tensor, dim=-1),"translationMean":torch.mean(tensor, dim=-1).mean(dim=0)}
    return result

def segmentAudio():
    frame_offset= 0
    num_frames= 1000
    filepath= ""
    waveform2, sample_rate2 = torchaudio.load(
        filepath, frame_offset=frame_offset, num_frames=num_frames
    )


class IWSLT2023(torch.utils.data.Dataset):
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torch.utils.datasets.IWSLT2023(
            "data", split=split, download=True
        )
        self.device = device
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)

dataset= IWSLT2023

references = []
transcriptions = []
translations = []
outputptobabilities = []
# Returns the last layer proabbliities of the model as a dict containing the decoded text and the segments and the language
asr_model.eval()



for audio, text in tqdm(dataset):
    # Load audio
    #audio = whisper.load_audio(audio)

#Pad and trim the audio to fit 30 seconds
    #audio = whisper.pad_or_trim(audio)

#this will generate the log-mel spectrogram of the given audio which will serve as an input to the whisper model 
    mel= whisper.log_mel_spectrogram(audio)

#Since the whisper is a multilingual ASR model, this will detect the spoken language. 
    _, probs = asr_model.detect_language(mel)
    print("Detected language is "+ " " + str({max(probs, key = probs.get)}))

#decode the audio 
    options = whisper.DecodingOptions()
    result = whisper.decode(asr_model,mel,options)

    # get the frist average log probability of the model for that aucio
    outputptobability = result[0].avg_logprob
    transcription = asr_model.transcribe(audio, **transcribe_options)["text"]
    #translation = asr_model.transcribe(audio, **translate_options)["text"]

    outputptobabilities.append(outputptobability)
    transcriptions.append(transcription)
    #translations.append(translation)
    references.append(text)