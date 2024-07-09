from sqlite3 import DatabaseError
import torch.utils
import torch.utils.data
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
import os
# try:
# import tensorflow  # required in Colab to avoid protobuf compatibility issues
# except ImportError:
#    pass

import yaml
import torch

#import whisper
import torchaudio

from tqdm import tqdm

#BASE = "$HOME"
#BASE = "/home/plantpalfynn/uni/BA-Thesis/code"
BASE = "/home/plantpalfynn/uni/BA/BA-Thesis/code"
# import ipywidgets as widgets
# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
options = dict(language="German", beam_size=5, best_of=5, dropout=0.3)
transcribe_options = dict(task="transcribe", **options)
# translate_options = dict(task="translate", **options)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    asr_model = asr_model.to(DEVICE)
# tar_path = download_asset("dataset/IWSLT23.tst2023.en-de.tar.gz")
# tar_item = "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/ted_13587.wav"
# with tarfile.open(tar_path, mode="r") as tarfile_:
#     fileobj = tarfile_.extractfile(tar_item)
#     waveform, sample_rate = torchaudio.load(fileobj)


# def transcriptionProbability(tensor, **options):
#     result = {
#         "translationProb": torch.nn.functional.softmax(tensor, dim=-1),
#         "translationMean": torch.mean(tensor, dim=-1).mean(dim=0),
#     }
#     return result


def getsegments():
    with open(
        BASE +"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "r",
    ) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data

def segmentAudio():
    #TODO make sectioning optional if the data is already sectioned and saved to seperate files
    dataset = getsegments()
    #print(dataset,"\n")
    sample_rate = 16000
    audiopath = BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    path = BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    resdataset = {
        "audiofile": [],
        #"waveform": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    if not os.path.exists(BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"):
        os.makedirs(BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/")
    for data in dataset:
        for i, seg in enumerate(dataset[data]):
            frame_offset = seg.get("offset") * 16000
            num_frames = seg.get("duration") * 16000
            waveform, sample_rate = torchaudio.load(
                audiopath + seg.get("wav"),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            path = BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"+ seg.get("wav") + str(i) + ".wav"
            if not os.path.exists(path):     
                torchaudio.save(path, waveform, sample_rate)

            resdataset["audiofile"].append(path)
            #resdataset["waveform"].append(waveform)
            resdataset["transcript"].append(seg.get("transcript"))
            # resdataset["audiofile"].append(seg.get("wav"))
            resdataset["translation"].append(seg.get("translation"))
            resdataset["timestamp"].append(
                (seg.get("offset"), seg.get("offset") + seg.get("duration"))
            )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    return resdataset


dataset = Dataset.from_dict(segmentAudio()).cast_column("audiofile", Audio())

# # Returns the last layer proabbliities of the model as a dict containing the decoded text and the segments and the language
# asr_model.eval()
result = {"sample": [], "audiofile": [], "timestamp": [], "res":[], "logits": [], "softmax": [], "outputptobability": [], "transcription": []}
#print(dataset[1]["audiofile"])
for i in tqdm(range(10)):
    sample = dataset[i]
    #for sample in tdqm(dataset):
    print(sample["transcript"])
    audio = waveform = sample["audiofile"]["array"]
    sample_rate = sample["audiofile"]['sampling_rate'] # alternatively set to 16000
    text = sample["transcript"]

    ########################################
    # Github whisper implementation things #
    ########################################
    # # this will generate the log-mel spectrogram of the given audio which will serve as an input to the whisper model
    # #mel = whisper.log_mel_spectrogram(audio)
    # Since the whisper is a multilingual ASR model, this will detect the spoken language.
    # #_, probs = asr_model.detect_language(mel)
    # #print("Detected language is " + " " + str({max(probs, key=probs.get)}))
    # # decode the audio
    # #options = whisper.DecodingOptions()
    # #result_trans = whisper.decode(asr_model, mel, options)
    # result["transcription"].append(
    #     asr_model.transcribe(audio, **transcribe_options)["text"]
    # )
    # translation = asr_model.transcribe(audio, **translate_options)["text"]

    #############################################
    # Huggingface whisper implementation things #
    #############################################

    # this will return the last layer probabilities of the model
    input= processor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    input_features= input.input_features.to(DEVICE)
    print(input_features, "\n result:")
    logits = asr_model.generate(input_features=input_features, output_scores=True)
    print(logits)
    #res = asr_model.generate(input_features=input_features)
    #print(type(res), "\n", res, "\n")

    #logits = asr_model(input_features).logits  # gets the last layer probabilities of the model
    #print(logits)
    # # get the frist average log probability of the model for that aucio
    result["audiofile"].append(sample["audiofile"])
    result["timestamp"].append(sample["timestamp"])
    result["logits"].append(logits)
    result["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
    result["outputptobability"].append(result[0].avg_logprob)
    #result["res"].append(res)
    result["sample"].append(sample)

print(result)
with open("result.txt", "w") as file:
    file.write(str(result))
# #segmentAudio()
file.close()