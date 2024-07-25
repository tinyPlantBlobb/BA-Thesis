from math import log
import os
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch

from tqdm import tqdm

class Result():
    audiofile=None
    timestamp= None
    runs = None
    ref = ""
    logits= None
    softmax= None
    generationoutput = None
    transcriptio= None

    def __init__(self, audiofile, timestamp, runs, ref):
        self.audiofile = audiofile
        self.timestamp = timestamp
        self.runs = runs["number"]
        self.ref = ref
        self.logits = runs["logits"]
        self.generationoutput = runs["generationoutput"]
        self.transcription = runs["transcription"]

    def __str__(self):
        return str(self.transcription)
    def __repr__(self):
        return

def TranslationProbability(data):
        #toptoken= data.generationoutput[i].scores
        print(len(data.generationoutput.scores[0]))
        prop= 1
        for j in range(len(data.generationoutput.scores)):
            toptoken= torch.argmax(torch.nn.functional.softmax(data.generationoutput.scores[j], dim=-1))
            
            prop= torch.log(data.generationoutput.scores[j][0][toptoken])+prop

  
def softmaxEntropy(data):
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

    prop= 1
    for j in range(len(data.generationoutput.scores)):
        prop= torch.sum(torch.nn.functional.log_softmax(data.generationoutput.scores[j], dim=-1)[0]*torch.nn.functional.log_softmax(data.generationoutput.scores[j], dim=-1)[0])+prop
    qeent= -(1/(len(data.generationoutput.scores[0]))*prop)
    return qeent

def sentStd(data):
    #TODO fix
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    print(len(data.generationoutput))

    sequence = []
    prop= 1
    for j in range(len(data.generationoutput.scores)):
        toptoken= torch.argmax(torch.nn.functional.softmax(data.generationoutput.scores[j], dim=-1))
        prop= torch.log(data.generationoutput.scores[j][0][toptoken])+prop 
        sequence.append(prop)
    qestd= np.std(np.array(sequence))
    
    return qestd


path = "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/dropoutresult70.pt"
t = torch.load(path, map_location=torch.device('cpu'))

# TranslationProbability(t)
# softmaxEntropy(t)
# print(t.transcription)
print(sentStd(t))