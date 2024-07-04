from cgitb import text
import torch.utils
import torch.utils.data
from datasets import load_dataset, Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

#try:
    #import tensorflow  # required in Colab to avoid protobuf compatibility issues
#except ImportError:
#    pass

import yaml
import torch

import whisper
import torchaudio

from scipy.io import wavfile
from tqdm import tqdm


#import ipywidgets as widgets
# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
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

def getsegments(): 
    with open("$HOME/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml", 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        
        return data
            #print(l, "\n", l.get("wav"))

def segmentAudio():
    dataset = getsegments()
    sample_rate= 16000
    audiopath= "$HOME/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.yaml"
    path = "$HOME/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    dataset= []
    for data in dataset:
        for i,seg in enumerate(data):
            frame_offset= seg.get("offset")*16000
            num_frames= seg.get("duration")*16000
            waveform, sample_rate = torchaudio.load(
                audiopath+seg.get("wav"), frame_offset=frame_offset, num_frames=num_frames
            )
            torchaudio.save(path+seg.get("wav")+i+"wav", waveform, sample_rate)
            
            dataset["audiofile"].append(path+seg.get("wav")+i+"wav")
            dataset["waveform"].append(waveform)
            dataset["sample_rate"]=sample_rate
            dataset["transcript"].append(seg.get("transcript"))
            dataset["audiofile"].append( seg.get("wav"))
            dataset["translation"].append(seg.get("translation"))
            dataset["timestamp"].append((seg.get("offset"),seg.get("offset")+seg.get("duration"))) #falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
            return dataset

dataset= Dataset.from_dict(segmentAudio()).cast_column("audiofile", Audio())

# class IWSLT2023(torch.utils.data.Dataset):
#     def __init__(self, split="test-clean", device=DEVICE):
#         self.dataset = segmentAudio()
        
#         self.device = device
    
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, item):
#         audio, sample_rate, text, _, _, _ = self.dataset[item]
#         assert sample_rate == 16000
#         audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
#         mel = whisper.log_mel_spectrogram(audio)
        
#         return (mel, text)



references = []
transcriptions = []
translations = []
outputptobabilities = []
# Returns the last layer proabbliities of the model as a dict containing the decoded text and the segments and the language
asr_model.eval()


for sample in tqdm(dataset):
    audio = waveform = sample["waveform"]
    sample_rate = sample["sample_rate"]
    text = sample["transcript"]

#this will generate the log-mel spectrogram of the given audio which will serve as an input to the whisper model 
    mel= whisper.log_mel_spectrogram(audio)

#Since the whisper is a multilingual ASR model, this will detect the spoken language. 
    _, probs = asr_model.detect_language(mel)
    print("Detected language is "+ " " + str({max(probs, key = probs.get)}))

#decode the audio 
    options = whisper.DecodingOptions()
    result = whisper.decode(asr_model,mel,options)
    input_features=  processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
    res= asr_model.generate(input_features.to("cuda"), output_scores=True)
    logits = asr_model().logits
    # get the frist average log probability of the model for that aucio
    outputptobability = result[0].avg_logprob
    transcription = asr_model.transcribe(audio, **transcribe_options)["text"]
    #translation = asr_model.transcribe(audio, **translate_options)["text"]

    outputptobabilities.append(outputptobability)
    transcriptions.append(transcription)
    #translations.append(translation)
    references.append(text)
