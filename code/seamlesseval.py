import re
import csv
from qeLogic import comet, pearsoncorr, variance, combo
import torch
modeltranslations =["" for i in range(2252)]
with open("reference.csv", "r") as ref:
    reader = csv.DictReader(ref, dialect="excel", fieldnames=["row", "reference transcript", "reference translation", "model trans", "qe"])
    for r in reader:

        modeltranslations[int(r["row"])%2252]=r["model trans"]


with open("workspaces/pfs5wor7/utqma-finals/seamlessdropoutresults.csv", "r") as res: 
    reader = csv.DictReader(res, dialect="excel", fieldnames=["row", "reference transcript", "reference translation", "model trans", "qe"])
    
    #output.sort(key=lambda x: int(x.split(",")[0])%2252)
    #splitted = [x.split(",")for x in output]
    #resdict={}
    #probs = [float(x.split(",")[-1]) for x in output]
    #print(splitted)
    qes=[[]for i in range(2252)]
    translationrefs=[i for i in range(2252)]
    transcriptrefs=[i for i in range(2252)]
    model=[[]for i in range(2252)]
    for r in reader:
        if r["qe"]!="qe":
            qes[int(r["row"])%2252].append(float(r["qe"]))
            translationrefs[int(r["row"])%2252]=r["reference translation"]
            transcriptrefs[int(r["row"])%2252]=r["reference transcript"]
            model[int(r["row"])%2252].append(r["model trans"])
    print(len(qes),len(qes[0]), len(transcriptrefs), len(translationrefs) )
    
    genqe= [sum(i[:30])/30 for i in qes]
    varqe=[variance(i[:30]) for i in qes]
    combo=combo(torch.as_tensor(genqe), torch.as_tensor(varqe)) 
    refscores = comet(transcriptrefs, translationrefs, modeltranslations).scores
    #print(type(qes), type(refscores))
    print("\n".join(list(map(lambda x: str(x[1])+" "+str(x[2])+" "+str(x[3])+" seperate "+" ".join(list(map(str, x[5])))+" "+x[4]+" ".join(x[0]), zip(model, genqe, varqe,refscores, translationrefs,qes)))), file=open("tst.txt", "w"))
    
    qescore = pearsoncorr(genqe, refscores)
    qevarscore = pearsoncorr(varqe, refscores)
    comescore = pearsoncorr(combo, refscores)
    print(qescore, qevarscore, comescore)
    
