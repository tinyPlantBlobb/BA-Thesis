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


def TranslationProbability(path):
    with open(path, "r") as file:
        data = torch.load(path, weights_only=False, map_location="cpu")
        #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        logits = data.logits    
        tp= []
        for i in range(len(logits)):
            #print(logits[i])
            sm= torch.nn.functional.softmax(logits[i], dim=-1)
            tp.append( (1/len(sm)) *torch.sum(torch.log(sm)))
        #print(tp)
        file.close()
        prop= 1
        print(len(data.generationoutput))
        for i in range(len(data.generationoutput)):
            #toptoken= data.generationoutput[i].scores
            print(len(data.generationoutput[i].scores[0]))
            prop= 1
            for j in range(len(data.generationoutput[i].scores)):
                toptoken= torch.argmax(torch.nn.functional.softmax(data.generationoutput[i].scores[j], dim=-1))
                print("toptoken", toptoken)
                print(data.generationoutput[i].scores[j][0][toptoken])
                prop= torch.log(data.generationoutput[i].scores[j][0][toptoken])+prop
                print(prop)
            print(i, (1/len(data.generationoutput[i].scores[0]))*prop)
            return (1/len(data.generationoutput[i].scores[0]))*prop

def softmaxEntropy(path):

    data = torch.load(path, weights_only=False)
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    logits = data["logits"]
    sm = data["softmax"]
    se = []
    for i in range(len(sm)):
        se.append(-1/len(sm[i])*torch.sum(sm[i]* torch.log(sm[i])))
    print(se)
    for i in range(len(data.generationoutput)):
        #toptoken= data.generationoutput[i].scores
        print(len(data.generationoutput[i].scores[0]))
        prop= 1
        for j in range(len(data.generationoutput[i].scores)):
            toptoken= torch.argmax(torch.nn.functional.softmax(data.generationoutput[i].scores[j], dim=-1))
            print("toptoken", toptoken)
            print(data.generationoutput[i].scores[j][0][toptoken])
            prop= torch.sum(data.generationoutput[i].scores[j][0]*torch.log(data.generationoutput[i].scores[j][0]))+prop
            print(prop)
        print(i, (-1/len(data.generationoutput[i].scores[0]))*prop)
        return (-1/len(data.generationoutput[i].scores[0]))*prop


path = "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/dropoutresult70.pt"
TranslationProbability(path)
#softmaxEntropy(path)
