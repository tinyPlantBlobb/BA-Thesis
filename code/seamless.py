import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
from transformers import SeamlessM4TForTextToText,  AutoProcessor as SeamlessProcessor
import os
import yaml
import torch
import torchaudio
from tqdm import tqdm
from qeLogic import TranslationProbability, softmaxEntropy, sentStd, Result

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4TForTextToText.from_pretrained("facebook/hf-seamless-m4t-medium")
processor = SeamlessProcessor.from_pretrained("facebook/seamless-m4t-v2-medium")



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
        )  # falls man noch die xmls rein matchen will: , "tranlate":seg.get("text"), "translation":seg.get("translation")
    print("finished iterating over elements")
    return resdataset


    

def run_inference(rank, world_size, dataset):
    # TODO make it work with the distributed data on the different gpus, aka figure out which rank to use and make it work
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    num_samples = 30
    elemdp = 20
    low = 50
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = low + rank*elemdp
    with torch.no_grad():
        for i in range(offset, offset+elemdp,1):
                
            all={
                "number":[],
                "tranlateion": [],
                "logits": [],
                #"softmax": [],
                "generationoutput": [],
            }

            sample = dataset[i]
            audio = waveform = sample["audiofile"]["array"]
            sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
            text = sample["tranlate"]
            ####################
            # dropout based shit#
            #####################
            for j in tqdm(range(num_samples)):
                #############################################
                # Huggingface whisper implementation things #
                #############################################
                all["number"].append(j)
                model.train()
                with torch.no_grad():
                    # this will return the last layer probabilities of the model
                    input = processor(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = input.input_features.to(rank)
                    res = model.generate(input_features=input_features,tgt_lang="deu", return_dict_in_generate=True, output_scores=True, output_logits=True, generate_speech=False)
                    logits = model(input_features, decoder_input_ids=res["sequences"]).logits  # gets the last layer probabilities of the model
                    all["logits"].append(logits)
                    all["generationoutput"].append(res)
                    dropoutresult= Result(audiofile=sample["audiofile"], timestamp=sample["timestamp"], runs=all,ref=text)
                torch.cuda.empty_cache()
            # with open(TEMPDIR + "/results/dropresult"+str(i)+".txt", "w") as file:
            #     file.write(str(dropoutresult["all"]["tranlateion"]))
            #     file.close()
            torch.save(dropoutresult, TEMPDIR + "/results/dropoutresult"+str(i)+".pt")
        del dropoutresult
        del sample

BASE = ""


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

class Result():
    audiofile=None
    timestamp= None
    runs = None
    ref = ""
    logits= None
    softmax= None
    generationoutput = None
    tranlation= None

    def __init__(self, audiofile, timestamp, runs, ref):
        self.audiofile = audiofile
        self.timestamp = timestamp
        self.runs = runs["number"]
        self.ref = ref
        self.logits = runs["logits"]
        self.generationoutput = runs["generationoutput"]
        self.tranlation = runs["tranlateion"]

    def __str__(self):
        return str(self.all["tranlateion"])
    def __repr__(self):
        return
