from tarfile import data_filter
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import os
import yaml
import torch
import torchaudio
from tqdm import tqdm

def loadinput():
    data = torch.load("")
    return data

def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
#deltalm=AutoModel.from_pretrained("deltalm-base.pt", trust_remote_code=True)

def main():
    world_size= 2
    mp.spawn(run_inference, args=(world_size,),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()

# load model