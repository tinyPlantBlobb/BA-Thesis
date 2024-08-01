from ast import List, Tuple
import csv
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
import tarfile
import torch
import torchaudio
from tqdm import tqdm
from qeLogic import getQE, writeCSV, Result, getAudios


def run_inference(rank, world_size, dataset):
    # TODO make it work with the distributed data on the different gpus, aka figure out which rank to use and make it work
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    num_samples = 30
    elemdp = 2
    low = 0
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = low + rank*elemdp
    with torch.no_grad():
        for i in range(offset, offset+elemdp,1):
            sample = dataset[i]
            all={
                "number":[],
                "transcription": [],
                "qetp": [],
                "qe-soft-ent"
                "generationoutput": [],
            }

            audio =sample["audiofile"]["array"]
            f= open(TEMPDIR + "/results/transcriptions"+str(i)+".csv", "w")   
            f.write("reference: " +sample["ref"]+ "\n")
            f.close()
            for j in tqdm(range(num_samples)):
                model.train()
                dpresults= []
                with torch.no_grad():
                    # this will return the last layer probabilities of the model
                    input = processor(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = input.input_features.to(rank)
                    res = model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
                    trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
                    # get the frist average log probability of the model for that aucio
                    # qe= TranslationProbability(res)
                    # qeent= softmaxEntropy(res)
                    # qestd= sentStd(res)
                    dpresults.append(res)
                    # getQE(res, dropout=True)
                    # all["qetp"].append(qe)
                    # all["qe-soft-ent"].append(qeent)
                    # all["qesent"].append(qestd)
                    all["generationoutput"].append(res)
                    all["transcription"].append(trans+"\n")
                    
                dropoutresult = getQE(res, dropouttrans = all["transcription"], dropout=True)
                torch.cuda.empty_cache()
            writeCSV(all["transcription"], TEMPDIR + "/results/transcriptions"+str(i)+".csv", sample["transcript"])
            torch.save(dropoutresult, TEMPDIR + "/results/dropoutresult"+str(i)+".pt")
        del dropoutresult
        del sample

BASE = ""


#dropout set to 0.1 based on the paper that uses the same dropout as during training
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en", dropout=0.1)
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    dataset = Dataset.from_dict(getAudios(TEMPDIR)).cast_column("audiofile", Audio())
    
    world_size= torch.cuda.device_count()
    torchrunrank= int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size,dataset),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()

