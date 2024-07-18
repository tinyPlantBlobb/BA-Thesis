import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

def getsegments(BASE):
    with open(
        BASE
        + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "r",
    ) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data


def segmentAudio(BASE):
    dataset = getsegments(BASE)
    sample_rate = 16000
    audiopath = BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    path = BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    if not os.path.exists(
        BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    ):
        os.makedirs(
            BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
        )
    for data in dataset:
        for i, seg in enumerate(dataset[data]):
            frame_offset = int(seg.get("offset") * 16000)
            num_frames = int(seg.get("duration") * 16000)
            waveform, sample_rate = torchaudio.load(
                audiopath + seg.get("wav"),
                frame_offset=frame_offset,
                num_frames=num_frames,
                format="wav",
                )

            path = (
                BASE
                + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
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

def getlogits(dataset, asr_model, processor, num_samples, rank, elements= 60, offset=0):

    if num_samples == 1:
       
        with torch.no_grad():
            for i in tqdm(range(offset, elements)):
                result = {
                    "audiofile": [],
                    "timestamp": [],
                    "ref": [],
                    "generationoutput": [],
                    "logits": [],
                    "softmax": [],
                    "transcription": [],
                }
                asr_model.eval()
                sample = dataset[i]
                # for sample in tdqm(dataset):
                #print(sample["transcript"])
                audio = waveform = sample["audiofile"]["array"]
                sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
                text = sample["transcript"]
                input = processor(audio, sampling_rate=16000, return_tensors="pt")
                input_features = input.input_features.to(rank)
                res = asr_model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
                print(res["sequences"])
                #############################################
                # Huggingface whisper implementation things #
                #############################################

                # this will return the last layer probabilities of the model

                logits = asr_model(
                    input_features, decoder_input_ids=res["sequences"]
                ).logits  # gets the last layer probabilities of the model
                trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
                # # get the frist average log probability of the model for that aucio
                result["audiofile"].append(sample["audiofile"])
                result["timestamp"].append(sample["timestamp"])
                result["logits"].append(logits)
                result["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
                result["generationoutput"].append(res)
                result["ref"].append(text)
                result["transcription"].append(trans+"\n")
                print(trans)
                print(text)
                # with open(TEMPDIR + "/results/result"+str(i)+".txt", "w") as file:
                #     file.write(str(result["transcription"]))
                #     file.close()
                torch.save(result, TEMPDIR + "/results/result"+str(i)+".pt")
                torch.cuda.empty_cache()

        return result
        
    elif num_samples == 30:
        
        with torch.no_grad():
            for i in range(offset, elements):
                dropoutresult = {
            "audiofile": [],
            "timestamp": [],
            "all":{"number":[],
                "transcription": [],
                "logits": [],
                "softmax": [],
                "generationoutput": [],
                },
            "ref": [],}
                sample = dataset[i]
                audio = waveform = sample["audiofile"]["array"]
                sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
                text = sample["transcript"]
                ####################
                # dropout based shit#
                #####################
                dropoutresult["audiofile"].append(sample["audiofile"])
                dropoutresult["timestamp"].append(sample["timestamp"])
                dropoutresult["ref"].append(text)
                for j in tqdm(range(num_samples)):
                    #############################################
                    # Huggingface whisper implementation things #
                    #############################################
                    dropoutresult["all"]["number"].append(j)
                    asr_model.train()
                    # this will return the last layer probabilities of the model
                    input = processor(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = input.input_features.to(rank)
                    res = asr_model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
                    logits = asr_model(input_features, decoder_input_ids=res["sequences"]).logits  # gets the last layer probabilities of the model
                    trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
                    # # get the frist average log probability of the model for that aucio

                    dropoutresult["all"]["logits"].append(logits)
                    dropoutresult["all"]["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
                    dropoutresult["all"]["generationoutput"].append(res)
                    dropoutresult["all"]["transcription"].append(trans+"\n")
                    
                    torch.cuda.empty_cache()
                with open(TEMPDIR + "/results/dropresult"+str(i)+".txt", "w") as file:
                    file.write(str(dropoutresult["all"]["transcription"]))
                    file.close()
                torch.save(dropoutresult, TEMPDIR + "/results/dropoutresult"+str(i)+".pt")

def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dataset = Dataset.from_dict(segmentAudio(BASE)).cast_column("audiofile", Audio())
    elemdp = 5
    low = 10
    if torch.distributed.get_rank() == 0:
        asr_model.to(rank)
        getlogits(dataset, asr_model, processor,1, rank, 30, 60)
    elif torch.distributed.get_rank() == 1:
        asr_model_drop.to(rank)
        getlogits(dataset, asr_model_drop, processor_drop, 30, rank, elemdp, low)
    elif torch.distributed.get_rank() == 2:
        asr_model_drop.to(rank)
        getlogits(dataset, asr_model_drop, processor_drop, 30, rank, elemdp, low + elemdp)
    elif torch.distributed.get_rank() == 3:
        asr_model_drop.to(rank)
        getlogits(dataset, asr_model_drop, processor_drop, 30, rank, elemdp, low+elemdp*2)
    # file.write(str(result))
    asr_model.generation_config.forced_decoder_ids = None



    

BASE = ""

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

#dropout set to 0.1 based on the paper that uses the same dropout as during training
asr_model_drop = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en", dropout=0.1)
processor_drop = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    world_size= 4
    mp.spawn(run_inference, args=(world_size,),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()
