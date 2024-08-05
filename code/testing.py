
import torch
from qeLogic import TranslationProbability, softmaxEntropy, sentStd, getQE


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

    def __init__(self, audiofile, timestamp, reference, transcription, modeldata, qualityestimate, dropoutdata=None, dropoutresults = None):
        self.audiofile = audiofile
        self.timestamp = timestamp
        self.ref = reference
        self.trans= transcription
        self.data= modeldata # result data of the model
        self.results= qualityestimate # Tuple of (qe, qeent, qestd)
        self.dropoutresults= dropoutresults
        self.dropoutdata= dropoutdata

    def __str__(self):
        return str(self.trans)
    def __repr__(self):
        return #"audiofile: "+str(self.audiofile)+ "\n" + "timestamp: "+str(self.timestamp)+ "\n" + "ref: "+str(self.ref)+ "\n" 


with open("/home/plantpalfynn/uni/BA/BA-Thesis/code/results/result0.pt", "rb") as f:
    data = torch.load(f)
    # Load data from file
    print(type(data.data))
    generationdata =data.data
    qeres= getQE(generationdata)