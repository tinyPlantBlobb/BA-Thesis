from ast import List, Tuple
from hmac import new
from pdb import run
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
import imp
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import os
import yaml
import tarfile
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
    # TODO make it work with the distributed data on the different gpus, aka figure out which rank to use and make it work
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    num_samples = 30
    elemdp = 20
    low = 50
    asr_model_drop.to(rank)
    asr_model_drop.generation_config.forced_decoder_ids = None
    offset = low + rank*elemdp
    with torch.no_grad():
        for i in range(offset, offset+elemdp,1):   
            all={
                "number":[],
                "transcription": [],
                "qetp": [],
                "qe-soft-ent"
                "generationoutput": [],
            }
            sample = dataset[i]
            audio = sample["audiofile"]["array"]
            text = sample["transcript"]
            ####################
            # dropout based shit#
            #####################
            for j in tqdm(range(num_samples)):
                #############################################
                # Huggingface whisper implementation things #
                #############################################
                all["number"].append(j)
                asr_model_drop.train()
                with torch.no_grad():
                    # this will return the last layer probabilities of the model
                    input = processor_drop(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = input.input_features.to(rank)
                    res = asr_model_drop.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
                    trans = processor_drop.batch_decode(res["sequences"], skip_special_tokens=True)[0]
                    # get the frist average log probability of the model for that aucio
                    qe= TranslationProbability(res)
                    qeent= softmaxEntropy(res)
                    qestd= sentStd(res)
                    
                    all["qetp"].append(qe)
                    all["qe-soft-ent"].append(qeent)
                    all["qesent"].append(qestd)
                    all["generationoutput"].append(res)
                    all["transcription"].append(trans+"\n")
                    dropoutresult= Result(audiofile=sample["audiofile"], timestamp=sample["timestamp"], runs=all,ref=text)
                torch.cuda.empty_cache()

            torch.save(dropoutresult, TEMPDIR + "/results/dropoutresult"+str(i)+".pt")
        del dropoutresult
        del sample

BASE = ""


#dropout set to 0.1 based on the paper that uses the same dropout as during training
asr_model_drop = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en", dropout=0.1)
processor_drop = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    dataset = Dataset.from_dict(readfromtar(BASE)).cast_column("audiofile", Audio())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    world_size= torch.cuda.device_count()
    torchrunrank= int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size,dataset),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()

