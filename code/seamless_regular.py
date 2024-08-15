import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset
from transformers import SeamlessM4Tv2ForTextToText,  AutoProcessor as SeamlessProcessor
import os
import torch
from tqdm import tqdm
from qeLogic import getQE,writeCSV, readCSV
# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessProcessor.from_pretrained("facebook/seamless-m4t-v2-large")


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank*((len(dataset))//world_size)
    #num=3
    num = (len(dataset))//(world_size)
    csv = []
    print("starting seamless regular on", num)
    with torch.no_grad():
        for i in tqdm(range(offset, offset+num,1)):
            model.eval()
            sample = dataset["train"][i]
            input = sample["transcription"]
            # alternatively set to 16000
            text = sample["reference"]
            text_input = processor(text=input,src_lang="eng", return_tensors="pt")
            
            text_input = text_input.to(rank)
            res = model.generate(**text_input, tgt_lang="deu", return_dict_in_generate=True, output_scores=True, output_logits=True)

            #############################################
            # Huggingface whisper implementation things #
            #############################################

            # this will return the last layer probabilities of the model

            #model(input_features, decoder_input_ids=res["sequences"]).logits  # gets the last layer probabilities of the model
            trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
            qe= getQE(res, dropout=False)
            torch.cuda.empty_cache()
            result = None
           # result = Result(sample["audiofile"],sample["timestamp"],sample["transcript"],trans,res,qe)
            torch.save(result, TEMPDIR + "/results/seamless_result"+str(i)+".pt")
            torch.cuda.empty_cache()
            
            print(trans, text)
            csv.append([i,text, trans, qe])
    output = [None for _ in range(world_size)]
    dist.gather_object(obj=csv, object_gather_list=output if dist.get_rank() == 0 else None,dst=0)
    if rank == 0:
        for i in range(len(output)):
            if i == 0:
                continue
            csv.extend(output[i])
        writeCSV(csv, TEMPDIR + "/results/seamlessfulltranscriptions.csv", dropout=False)

BASE = ""

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    print(TEMPDIR + "/results/fulltranscriptions.csv")
    dataset = load_dataset("csv",data_files=TEMPDIR + "/results/fulltranscriptions.csv")
    world_size= torch.cuda.device_count()
    torchrunrank= int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    smp = mp.get_context('spawn')
    q   = smp.SimpleQueue()
    q.put([["sample","reference","reference"]])
    mp.spawn(run_inference, args=(world_size,dataset),nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

