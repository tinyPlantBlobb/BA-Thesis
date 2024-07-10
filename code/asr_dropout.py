import torch.utils
import torch.utils.data
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import os
import yaml
import torch
import torchaudio
from tqdm import tqdm

BASE = "$HOME"
#BASE = "/home/plantpalfynn/uni/BA-Thesis/code"
#BASE = "/home/plantpalfynn/uni/BA/BA-Thesis/code"

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

#dropout set to 0.1 based on the paper that uses the same dropout as during training
asr_model_drop = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en", dropout=0.1)
processor_drop = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
options = dict(language="German", beam_size=5, best_of=5, dropout=0.3)
transcribe_options = dict(task="transcribe", **options)
# translate_options = dict(task="translate", **options)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    asr_model = asr_model.to(DEVICE)
    asr_model_drop = asr_model_drop.to(DEVICE)

asr_model.generation_config.forced_decoder_ids = None


def getsegments():
    with open(
        BASE
        + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "r",
    ) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data


def segmentAudio():
    dataset = getsegments()
    # print(dataset,"\n")
    sample_rate = 16000
    audiopath = BASE + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    path = BASE + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    resdataset = {
        "audiofile": [],
        # "waveform": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    if not os.path.exists(
        BASE + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    ):
        os.makedirs(
            BASE + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
        )
    for data in dataset:
        for i, seg in enumerate(dataset[data]):
            frame_offset = seg.get("offset") * 16000
            num_frames = seg.get("duration") * 16000
            waveform, sample_rate = torchaudio.load(
                audiopath + seg.get("wav"),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            path = (
                BASE
                + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
                + seg.get("wav")
                + str(i)
                + ".wav"
            )
            if not os.path.exists(path):
                torchaudio.save(path, waveform, sample_rate)

            resdataset["audiofile"].append(path)
            # resdataset["waveform"].append(waveform)
            resdataset["transcript"].append(seg.get("transcript"))
            # resdataset["audiofile"].append(seg.get("wav"))
            resdataset["translation"].append(seg.get("translation"))
            resdataset["timestamp"].append(
                (seg.get("offset"), seg.get("offset") + seg.get("duration"))
            )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    return resdataset


dataset = Dataset.from_dict(segmentAudio()).cast_column("audiofile", Audio())

# # Returns the last layer proabbliities of the model as a dict containing the decoded text and the segments and the language

result = {
    #"sample": [],
    "audiofile": [],
    "timestamp": [],
    "ref": [],
    "logits": [],
    "softmax": [],
    #    "outputptobability": [],
    "transcription": [],
}
dropoutresult = {
    #"sample": [],
    "audiofile": [],
    "timestamp": [],
    "all":{"number":[],
           "transcription": [],
           "logits": [],
            "softmax": [],
        },
    
    "ref": [],
    #    "outputptobability": [],
    
}
# print(dataset[1]["audiofile"])
with torch.no_grad():
    for i in tqdm(range(10)):
        asr_model.eval()
        sample = dataset[i]
        # for sample in tdqm(dataset):
        #print(sample["transcript"])
        audio = waveform = sample["audiofile"]["array"]
        sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
        text = sample["transcript"]

        #############################################
        # Huggingface whisper implementation things #
        #############################################

        # this will return the last layer probabilities of the model
        input = processor_drop(audio, sampling_rate=16000, return_tensors="pt")
        input_features = input.input_features.to(DEVICE)
        # logitsmaybe = asr_model.generate(input_features=input_features, output_scores=True)
        # print(logitsmaybe)
        res = asr_model.generate(input_features=input_features)
        
        logits = asr_model(
            input_features, decoder_input_ids=res
        ).logits  # gets the last layer probabilities of the model
        trans = processor.batch_decode(res, skip_special_tokens=True)[0]
        # # get the frist average log probability of the model for that aucio
        result["audiofile"].append(sample["audiofile"])
        result["timestamp"].append(sample["timestamp"])
        result["logits"].append(logits)
        result["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
        # result["outputptobability"].append(result[0].avg_logprob)
        result["ref"].append(text)
        #result["sample"].append(sample)
        result["transcription"].append(trans)
        print(trans)
        print(text)
        torch.cuda.empty_cache()

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


        ####################
        # dropout based shit#
        #####################
        dropoutresult["audiofile"].append(sample["audiofile"])
        dropoutresult["timestamp"].append(sample["timestamp"])
        dropoutresult["ref"].append(text)
        for j in tqdm(range(30)):
            #############################################
            # Huggingface whisper implementation things #
            #############################################
            dropoutresult["all"]["number"].append(j)
            asr_model_drop.train()
            # this will return the last layer probabilities of the model
            input = processor_drop(audio, sampling_rate=16000, return_tensors="pt")
            input_features = input.input_features.to(DEVICE)
            # logitsmaybe = asr_model.generate(input_features=input_features, output_scores=True)
            # print(logitsmaybe)
            res = asr_model_drop.generate(input_features=input_features)
            
            logits = asr_model_drop(
                input_features, decoder_input_ids=res
            ).logits  # gets the last layer probabilities of the model
            trans = processor_drop.batch_decode(res, skip_special_tokens=True)[0]
            # # get the frist average log probability of the model for that aucio
            
            dropoutresult["all"]["logits"].append(logits)
            dropoutresult["all"]["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
            # result["outputptobability"].append(result[0].avg_logprob)
            
            #result["sample"].append(sample)
            dropoutresult["all"]["transcription"].append(trans)

            torch.cuda.empty_cache()


with open("result.yaml", "w") as file:
    # TODO better output file format, maybe a yaml?
    file.write(str(result["transcription"]))
    file.write(str(dropoutresult["all"]["transcription"]))
# file.write(str(result))
file.close()
torch.save(result, "result.pt")
torch.save(dropoutresult, "dropoutresult.pt")