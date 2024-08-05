import os
import csv
import torch.distributed
import torch.utils
import torch.utils.data
import numpy as np
import torch
#import evaluate 
import yaml
import torchaudio
import pdb; pdb.set_trace()

def getAudios(TEMPDIR):
    print("starting reading from tar")
    with open(TEMPDIR+"/data/IWSLT.TED.tst2023.en-de.matched.yaml") as matched:
        data = yaml.load(matched, Loader=yaml.FullLoader)
        matched.close()
    print("closed tar")

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
        torchaudio.load(t.path)
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
    prop= 0
    for j in range(len(data.scores)):
        toptoken= torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        toptokenprob = torch.log_softmax(data.scores[j][0], dim=-1)[toptoken]
        prop= toptokenprob+prop
    return np.divide(prop.cpu().numpy(),len(data.scores[0]))

  
def softmaxEntropy(data):
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    prop= 0
    for j in range(len(data.scores)):
        softmaxed = data.scores[j]
        
        print("softmax", softmaxed[0], type(softmaxed[0]))
        for i in range(len(data.scores[j])):
            for k in range(len(data.scores[j][i])):
                #breakpoint()
                print("softmaxed",softmaxed[i][k].item(), torch.mul(softmaxed[i][k],torch.log(softmaxed[i][k])),type(torch.mul(softmaxed[i][k],torch.log(softmaxed[i][k]))))
                prop= torch.mul(softmaxed[i][k],torch.log(softmaxed[i][k]))+prop
                #breakpoint()
                print("prop",prop)
        print("result",prop, type(prop))
    qeent= -np.divide(prop.cpu().numpy(), (len(data.scores[0])))
    return qeent

def sentStd(data):
    #TODO fix
    #Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    sequence = []
    prop= 0
    for j in range(len(data.scores)):
        toptoken= torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        prop= torch.log_softmax(data.scores[j][0][toptoken], dim=-1)+prop 
        sequence.append(prop.cpu())
    print(sequence)
    qestd= np.std(np.array(sequence))
    
    return qestd

def writeCSV(results, path, dropout=False):
    if dropout:
        with open(path, "a", newline='') as f:
            
            writer = csv.writer(f, dialect='excel')
            #writer.writerow(["reference", "transcriptions"])
            writer.writerows(results)
    else:
        with open(path, "w", newline='') as f:
            writer = csv.writer(f, dialect='excel')
            #writer.writerow(["reference", "transcription"])
            writer.writerows(results)

def readCSV(path):
    with open(path, 'r',newline='') as f:
        reader = csv.reader(f, dialect='excel')
        data = {rows[0]:(rows[1], rows[2]) for rows in reader}
    return data

def variance(data):
    return np.var(data)

def combo(tp, var):
    return (1-np.divide(tp. var))

def lexsim(transhypo):
    #meteor = evaluate.load('meteor')
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
        qestd= sentStd(data)
        res = (qe, qeent, qestd)
    print(res)
    return res
    


# TranslationProbability(t)
# softmaxEntropy(t)
# print(t.transcription)
