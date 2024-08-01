import os
import csv
import torch.distributed
import torch.utils
import torch.utils.data
import numpy as np
import torch
import evaluate 
import yaml
import torchaudio


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

def getAudios(TEMPDIR):
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
        )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    print("finished iterating over elements")
    return resdataset


def TranslationProbability(data):
        #toptoken= data[i].scores
        prop= 1
        for j in range(len(data.scores)):
            toptoken= torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
            toptokenprob = torch.log_softmax(data.scores[j][0])[toptoken]
            print(toptokenprob)
            prop= toptokenprob+prop

  
def softmaxEntropy(data):
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    prop= 1
    for j in range(len(data.scores)):
        prop= torch.sum(torch.nn.functional.log_softmax(data.scores[j], dim=-1)[0]*torch.nn.functional.log_softmax(data.scores[j], dim=-1)[0])+prop
    qeent= -(1/(len(data.scores[0]))*prop)
    return qeent

def sentStd(data):
    #TODO fix
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    print(len(data))

    sequence = []
    prop= 1
    for j in range(len(data.scores)):
        toptoken= torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        prop= torch.log(data.scores[j][0][toptoken])+prop 
        sequence.append(prop)
    qestd= np.std(np.array(sequence))
    
    return qestd

def writeCSV(results, path, refence=None, dropout=False):
    if dropout:
        with open(path, "a", newline='') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(refence, results)
    else:
        with open(path, "w", newline='') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(["reference", "transcription"])
            for i in range(len(results)):
                writer.writerow([refence, results])

def readCSV(path):
    with open(path, 'r',newline='') as f:
        reader = csv.reader(f, dialect='excel')
        data = list(reader)
    return data

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
        for i in range(len(data)):
            qe= TranslationProbability(data)
            qevar= variance(data)
            com = combo(qe, qevar)
            lex = lexsim(dropouttrans)
        res =(qe, qevar, com)
    else:
        qe= TranslationProbability(data)
        qeent= softmaxEntropy(data)
        #qestd= sentStd(data)
        res =(qe, qeent)
    return res
    


# TranslationProbability(t)
# softmaxEntropy(t)
# print(t.transcription)
