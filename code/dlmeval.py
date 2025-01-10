import re
import csv
import qeLogic
import numpy as np
import torch
from itertools import groupby
BASE = "/pfs/work7/workspace/scratch/utqma-finals/results-25078050/"
#BASE = "~/uni/BA/BA-Thesis/code/"
ref = []
transcriptscore=[]
transcript=[]
reftranscript=[]
transcriptmean=[]
with open("/pfs/work7/workspace/scratch/utqma-finals/results-25081042/dropoutfulltranscriptions.csv","r") as trnsscore:
    reader = csv.DictReader(trnsscore, dialect="excel")
    for i in reader:
        transcriptscore.append(float(i["qe"].strip("(){[]}").split(",")[0]))
        transcriptmean.append(float(i["qe"].strip("(){[]}").split(",")[1]))
        transcript.append(i["regulartransaltion"])
        reftranscript.append(i["reference transcript"])
with open("results/dlmresults.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        [
            "non-droput translation",
            "non-dropout source sentence",
            "non-dropout translation probability",
            "non-dropout softmax entropy",
            "non-dropout variance",
            "Target",
            "Probabilities",
            # "Hypothesis",
            "Source Sentences",
            "Detokenized Hypothesis",
            "Vocabulary scores",
            "qe",
            "dropout translation probability",
            "dropout transation mean",
        ]
    )
    dptpqe = [0.0] * 2255
    dpvar = [0.0] * 2255
    row = []
    with open(BASE+"generate-test.txt", "r") as f:
        lines = f.readlines()
        output = [i.strip() for i in lines if re.search(r"^[TPHSDV]-\d.*", i)]
        output.sort(key=lambda x: int(re.search(r"^[TPDHSV]-(\d+)", x).group(1)))
        target = [
            i.split("\t")[1] if len(i.split("\t")) > 1 else " "
            for i in output
            if re.search(r"^T-\d.*", i)
        ]
        
        dptpqe = dptpqe
        dpvar = dpvar
        probabilities = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"^P-\d.*", i) and len(i.split("\t")) > 1
        ]

        hypothesis = [
            i.split("\t")[1:]
            for i in output
            if re.search(r"^H-\d.*", i) and len(i.split("\t")) > 1
        ]

        srcsentences = [
            i.split("\t")[1] if len(i.split("\t")) > 1 else " "
            for i in output
            if re.search(r"^S-\d.*", i)
        ]

        detokenizedhypothesis = [
            i.split("\t")[1:] for i in output if re.search(r"^D-\d.*", i)
        ]

        # print(detokenizedhypothesis)
        vocabscores = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"V-\d.*", i)
        ]

        softmaxentropy = [sum(i) / len(i) for i in vocabscores][:2255]
        variances = [qeLogic.stddiv(i) for i in probabilities][:2255]
        probs =[float(i[0]) for i in detokenizedhypothesis][:2255]

        # with open("dlmprint.txt", "w") as pri:
        # pri.write("\n".join([k[1] for k in detokenizedhypothesis]))

        line = [
            [
                detokenizedhypothesis[i][1],
                srcsentences[i],
                target[i],
                detokenizedhypothesis[i][0],
                softmaxentropy[i],
                variances[i],
                [detokenizedhypothesis[i][1]],
                [probabilities[i]],
                [float(detokenizedhypothesis[i][0])],
                sum(probabilities[i][1:]) / len(probabilities[i][1:]),
                qeLogic.variance(probabilities[i]),
                [softmaxentropy[i]]
            ]
            for i in range(2255)
        ]

        row.extend(line)
        #csvwriter.writerows(row)
        hypothesis = [i[1] for i in detokenizedhypothesis]
        #print(len(hypothesis), len(target), len(srcsentences))
        ref = qeLogic.cometscore(srcsentences, hypothesis, target)
        ref=ref["scores"]
        print(len(ref))
        ref = ref[:2255]
        #probabilitycorrelation = qeLogic.pearsoncorr(probs, ref)
        #softmaxcorrelation = qeLogic.pearsoncorr(softmaxentropy, ref)
        #varcorrelation = qeLogic.pearsoncorr(variances, ref)
        #print(
        #    "probabilitycorrelation",
        #    probabilitycorrelation,
        #    "softmaxcorrelation",
        #    softmaxcorrelation,
        #    "varcorrelation",
        #    varcorrelation,
        #)
    # for i in range(1, 30):
    with open(BASE+"dropout-test.txt", "r") as f:
        lines = f.readlines()
        #print(len(lines))
        output = [i.strip() for i in lines if re.search(r"^[VTPHSD]-\d.*", i)]

        output.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x).group(1))%2255)
        
        #for i in range(2255): 
            #print(len([j for j in output if int(re.search(r"^[TPDVHS]-(\d+)",j).group(1))%2255==i]))
        #keys=[]
        #groups = []
        #alllinesdict={1:[]}

        #for key, group in groupby(output, lambda x:(int(re.search(r"^[TDPHSV]-(\d+)",x).group(1))%2252)):
        #    if key:
        #        keys.append(key)
        #        groups.append(list(group))
        #        dictelem = alllinesdict.get(key,[])
        #        print(key,dictelem)
        #        alllinesdict[key] =dictelem.append(list(group))
        #
        #print(list(groups), file=open("tst","w"))
        #        #print
        #     print([i.split("\t")[0]for i in output], file=open("out.txt","w"))
        target = [j.split("\t")[1] if len(j.split("\t"))>1 else " " for j in output if re.search(r"^T-\d+.*", j)]
        #print([i.split("\t") for i in re.findall(r"^T-\d+.*", i) ])
        test =[i.split("\t")[0] for i in output if re.search(r"^T-\d+", i)]
#
        probabilities = [
         list(map(float, i.split("\t")[1].split(" ")))
         for i in output
         if re.search(r"^P-\d.*", i)
     ]
        hypothesis = [i.split("\t")[1:] for i in output if re.search(r"^H-\d.*", i) and len(i.split("\t"))>1]
        srcsentences = [
             i.split("\t")[1:] if len(i.split("\t"))>1 else " " for i in output if re.search(r"^S-\d.*", i)
         ]
        detokenizedhypothesis = [
             i.split("\t")[1:] for i in output if re.search(r"^D-\d.*", i) and len(i.split("\t"))>1
        ]
        vocabscores = [
             list(map(float, i.split("\t")[1].split(" ")))
             for i in output
             if re.search(r"V-\d.*", i)
        ]
         #probabilities.sort(key=lambda x: int(re.search(r"^[TPHDSV]-(\d+)",x)))
         #hypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
         #srcsentences.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
         #detokenizedhypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
         #target.sort(key=lambda x:int(re.search(r"^[TDPHSV]-(\d+)",x)))
         
        softmaxentropydp = [sum(i) / len(i) for i in vocabscores]
    
        for i in range(2255):
            #print(j)
            for j in range(30):
        
                #print(i,j,i*j, i*j+j, detokenizedhypothesis[i*j], file=open("tst","a"))
                dptpqe[i] += float(detokenizedhypothesis[30*i+j][0])
                #row[i][5].append(target[j])
                #print(i,j, 30*i+j)
                row[i][6].append(detokenizedhypothesis[30*i+j][1])
                row[i][8].append(float(detokenizedhypothesis[30*i+j][0]))
                row[i][11].append(softmaxentropydp[30*i+j])
                #print(len(target), len(row), )
                #print(i, j, target[j*i])
                #row[i][7].extend(probabilities[i*j:i*j+30])
            #row[i][7].append(srcsentences[j])
            #row[i][10].append(-sum(vocabscores[j*i]) / len(vocabscores[j*i]))
        #[print(row[i][8]) for i in range(2252)]
        csvwriter.writerows(row)
    dpvar = [qeLogic.variance(row[i][8]).cpu().numpy().item() for i in range(2255)]
    dptqe =[i/30 for i in dptpqe]
    #print(type(probs), probs[0],type(transcriptscore[0]))
    uniscore=[i[0]*i[1] for i in zip(transcriptscore, probs)]
    uniscore2=[i[0]*i[1] for i in zip(transcriptmean, probs)]
    
    entropydp=[sum(row[i][11])/30 for i in range(2255)]
    entropydpvar=[qeLogic.variance(row[i][11]) for i in range(2255)]
    wer=qeLogic.worderror(transcript, reftranscript)
    uniscoreref=[i[0]*(1-i[1]/max(wer)) for i in zip(ref, wer)]
    #print(len(ref), len(probs), dptpqe)
    combo = [1-dptqe[i]/dpvar[i] for i in range(len(dpvar))]
    probabilitycorrelation = qeLogic.pearsoncorr(probs, ref)
    softmaxcorrelation = qeLogic.pearsoncorr(softmaxentropy, ref)
    varcorrelation = qeLogic.pearsoncorr(variances, ref)
    dpprob = qeLogic.pearsoncorr(dptqe, ref)
    dpvarcorr = qeLogic.pearsoncorr(dpvar,ref)
    dpcombo = qeLogic.pearsoncorr(combo, ref)

    uniscoreres=qeLogic.pearsoncorr(uniscore, uniscoreref)
    uniscore2res=qeLogic.pearsoncorr(uniscore2, uniscoreref)
    entrpydpres=qeLogic.pearsoncorr(entropydp, ref)
    entropydpvarres=qeLogic.pearsoncorr(entropydpvar, ref)

    print(
         "probabilitycorrelation ",
         probabilitycorrelation,
         "\n softmaxcorrelation ",
         softmaxcorrelation,
         "\n std correlation ",
         varcorrelation,
         "\n dp prob corr ",
         dpprob,
         "\n dp var corr ",
         dpvarcorr, 
         "\n dpcombo ", dpcombo,
         "\nentropy dropout ", entrpydpres,
         "\nentropy variance ",entropydpvarres,
         "\n uniscores ", uniscoreres, 
         "\n res 2: ", uniscore2res, 

    )
    for i in range(2255):
        print(
            probs[i],
            "\t",
            softmaxentropy[i],
            "\t",
            variances[i],
            "\t",
            dptqe[i],
            "\t",
            dpvar[i],
            "\t",
            combo[i],
            "\t",
            uniscore[i],"\t",
            uniscore2[i], "\t",
            ref[i],
            "\t", wer[i], 
            "\t",uniscoreref[i],
            file=open("dlmallscores.txt", "a")),"\t"
    
    for a in range(100):
        uniscore=[(1-a/100)*i[0]+(a/100)*i[1] for i in zip(transcriptscore, probs)]
        uniscore2=[(1-a/100)*i[0]+(a/100)*i[1] for i in zip(transcriptmean, probs)]
        uniscoreb=[float(pow(i[1],(1-a/100)))+float(pow(i[0],(a/100))) for i in zip(transcriptscore, probs)]
        uniscoreb2=[float(pow(i[1],(1-a/100)))+float(pow(i[0],(a/100))) for i in zip(transcriptmean, probs)]
        print(a, "\t", qeLogic.pearsoncorr(uniscore, uniscoreref)["pearsonr"],"\t", qeLogic.pearsoncorr(uniscore2, uniscoreref)["pearsonr"],"\t", qeLogic.pearsoncorr(uniscoreb, uniscoreref)["pearsonr"],"\t", qeLogic.pearsoncorr(uniscoreb2, uniscoreref)["pearsonr"], file=open("dlmuniscores.txt", "a"))
