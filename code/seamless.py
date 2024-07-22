
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
from transformers import AutoTokenizer, AutoModel
import os
import yaml
import torch
import torchaudio
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
model = AutoModel.from_pretrained("facebook/seamless-m4t-v2-large")