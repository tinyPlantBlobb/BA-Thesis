from math import log
import os
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch

from tqdm import tqdm
def TranslationProbability(path):
    with open(path, "r") as file:
        data = torch.load(path, weights_only=False)
        #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        logits = data["logits"]
        tp= []
        for i in range(len(logits)):
            tp.append( 1/len(logits[i]) *torch.log(logits[i]))
        print(tp)
        file.close()

def softmaxEntropy(path):
    with open(path, "r") as file:
        data = torch.load(path, weights_only=False)
        #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        logits = data["logits"]
        sm = data["softmax"]
        se = []
        for i in range(len(sm)):
            se.append(-1/len(sm[i])*torch.sum(sm[i]* torch.log(sm[i])))
        print(se)
        file.close()

path = "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/result206.pt"
TranslationProbability(path)
softmaxEntropy(path)
