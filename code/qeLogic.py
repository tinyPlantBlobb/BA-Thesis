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
import evaluate 
from tqdm import tqdm


class Result():
    audiofile=None
    timestamp= None
    runs = None
    ref = None
    trans= None
    data= None # result of the model
    results= None # Tuple of (qe, qeent, qestd)
    dropoutdata= None # result of the model for all droutout runs list of tuples
    dropoutresults= None # list of Tuple of (qe, var, lex-simm)

    def __init__(self, audiofile, timestamp, ref, trans, data, results, dropoutdata=None, dropoutresults = None):
        self.audiofile = audiofile
        self.timestamp = timestamp
        self.ref = ref
        self.trans= trans
        self.data= data
        self.results= results
        self.dropoutresults= dropoutresults
        self.dropoutdata= dropoutdata


    def __str__(self):
        return str(self.trans)
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

def variance(data):
    return np.var(data)

def combo(tp, var):
    return (1-np.divide(tp. var))

def lexsim(transhypo):
    meteor = evaluate.load('meteor')
    #TODO write code for the simmilarity with the help of meteor 
    
    return 0

def getQE(data, dropout=False, dropouttrans=None):
    if dropout:
        for i in range(len(data.generationoutput)):
            qe= TranslationProbability(data)
            qevar= variance(data)
            com = combo(qe, qevar)
            lex = lexsim(dropouttrans)
        res =(qe, var, com)
    else:
        qe= TranslationProbability(data)
        qeent= softmaxEntropy(data)
        qestd= sentStd(data)
        res =(qe, qeent, qestd)
    return res
    
path = "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/dropoutresult70.pt"
t = torch.load(path, map_location=torch.device('cpu'))

# TranslationProbability(t)
# softmaxEntropy(t)
# print(t.transcription)
print(sentStd(t))