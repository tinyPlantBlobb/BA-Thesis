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
from qeLogic import TranslationProbability, softmaxEntropy, sentStd, Result
def readfromtar(BASE):
    print("starting reading from tar")
    with open(TEMPDIR+"/data/IWSLT.TED.tst2023.en-de.matched.yaml") as matched:
        data = yaml.load(matched, Loader=yaml.FullLoader)
        matched.close()
    print("closed tar")
    sample_rate = 16000
    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    print("starting iterating over tar elements")
    for t in os.scandir(TEMPDIR+"/data"):
        if t.name == "IWSLT.TED.tst2023.en-de.matched.yaml":
            continue
        tedwav = t.name.split(".")[0]
        segment= int(t.name.split(".")[1][3:]) 
        seg = data[tedwav+".wav"][segment]
        waveform, samplerate = torchaudio.load(t.path)
        resdataset["audiofile"].append(t.path)
        resdataset["transcript"].append(seg.get("transcript"))
        resdataset["translation"].append(seg.get("translation"))
        resdataset["timestamp"].append(
            (seg.get("offset"), seg.get("offset") + seg.get("duration"))
        )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    print("finished iterating over elements")
    return resdataset


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    asr_model.to(rank)
    asr_model.generation_config.forced_decoder_ids = None

    offset = 180 + rank*((len(dataset)-180)//world_size)
    with torch.no_grad():
        for i in tqdm(range(offset, offset+30,1)):
            result = {
                "audiofile": [],
                "timestamp": [],
                "ref": [],
                "generationoutput": [],
                "logits": [],
                "softmax": [],
                "transcription": [],
                "qe": [],
                "qeent": [],
            }
            asr_model.eval()
            sample = dataset[i]
            # for sample in tdqm(dataset):
            #print(sample["transcript"])
            audio =sample["audiofile"]["array"]
            sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
            text = sample["transcript"]
            input = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = input.input_features.to(rank)
            res = asr_model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
            #############################################
            # Huggingface whisper implementation things #
            #############################################

            # this will return the last layer probabilities of the model

            logits = asr_model(
                input_features, decoder_input_ids=res["sequences"]
            ).logits  # gets the last layer probabilities of the model
            trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
            qe = TranslationProbability(res)
            qeent = softmaxEntropy(res)
            qesent = sentStd(res)


            # # get the frist average log probability of the model for that aucio
            result["qe"].append(qe)
            result["qeent"].append(qeent)
            result["qesent"].append(qesent)
            result["audiofile"].append(sample["audiofile"])
            result["timestamp"].append(sample["timestamp"])
            result["logits"].append(logits)
            result["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
            result["generationoutput"].append(res)
            result["ref"].append(text)
            result["transcription"].append(trans+"\n")
            torch.save(result, TEMPDIR + "/results/result"+str(i)+".pt")
            torch.cuda.empty_cache()

BASE = ""

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    dataset = Dataset.from_dict(readfromtar(BASE)).cast_column("audiofile", Audio())
    world_size= torch.cuda.device_count()
    torchrunrank= int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size,dataset),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()
