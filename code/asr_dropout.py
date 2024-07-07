from sqlite3 import DatabaseError
import torch.utils
import torch.utils.data
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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
BASE = "/home/plantpalfynn/uni/BA-Thesis/code"

# import ipywidgets as widgets
# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
options = dict(language="German", beam_size=5, best_of=5, dropout=0.3)
transcribe_options = dict(task="transcribe", **options)
# translate_options = dict(task="translate", **options)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# tar_path = download_asset("dataset/IWSLT23.tst2023.en-de.tar.gz")
# tar_item = "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/ted_13587.wav"
# with tarfile.open(tar_path, mode="r") as tarfile_:
#     fileobj = tarfile_.extractfile(tar_item)
#     waveform, sample_rate = torchaudio.load(fileobj)


def transcriptionProbability(tensor, **options):
    result = {
        "translationProb": torch.nn.functional.softmax(tensor, dim=-1),
        "translationMean": torch.mean(tensor, dim=-1).mean(dim=0),
    }
    return result


def getsegments():
    with open(
        BASE +"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "r",
    ) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

        return data
        # print(l, "\n", l.get("wav"))


def segmentAudio():
    #TODO make sectioning optional if the data is already sectioned and saved to seperate files
    dataset = getsegments()
    print(dataset,"\n")
    sample_rate = 16000
    audiopath = BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    path = BASE+"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    resdataset = {
        "audiofile": [],
        "waveform": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
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
            torchaudio.save(path, waveform, sample_rate)

            resdataset["audiofile"].append(path)
            resdataset["waveform"].append(waveform)
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
# result = {}

# for sample in tqdm(dataset[:10]):
#     audio = waveform = sample["waveform"]
#     sample_rate = sample["sample_rate"]
#     text = sample["transcript"]

#     # this will generate the log-mel spectrogram of the given audio which will serve as an input to the whisper model
#     #mel = whisper.log_mel_spectrogram(audio)

#     # Since the whisper is a multilingual ASR model, this will detect the spoken language.
#     #_, probs = asr_model.detect_language(mel)
#     #print("Detected language is " + " " + str({max(probs, key=probs.get)}))

#     # decode the audio
#     #options = whisper.DecodingOptions()
#     #result_trans = whisper.decode(asr_model, mel, options)

#     # this will return the last layer probabilities of the model
#     input_features = processor(
#         waveform, sampling_rate=16000, return_tensors="pt"
#     ).input_features
#     res = asr_model.generate(input_features.to("cuda"), output_scores=True)
#     logits = asr_model(res).logits  # gets the last layer probabilities of the model
#     # get the frist average log probability of the model for that aucio
#     result["audiofile"].append(sample["audiofile"])
#     result["timestamp"].append(sample["timestamp"])
#     result["logits"].append(logits)
#     result["outputptobability"].append(result[0].avg_logprob)
#     result["transcription"].append(
#         asr_model.transcribe(audio, **transcribe_options)["text"]
#     )
#     # translation = asr_model.transcribe(audio, **translate_options)["text"]
#     print(result)
segmentAudio()