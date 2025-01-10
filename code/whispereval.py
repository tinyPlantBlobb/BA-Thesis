import re
import csv
import math
from qeLogic import worderror, pearsoncorr, variance
import torch
import werpy
with open("dropoutfulltranscriptions.csv", "r") as f:

        row = [
            "row",
            "reference transcript",
            "reference translation",
            "qe",
        ]
        row.extend(["transcript probability " + str(i) for i in range(30)])
        row.extend(["transcript " + str(i) for i in range(30)])
        probabilities = ["transcript probability " + str(i) for i in range(30)]
        csvtranscripts = ["transcript " + str(i) for i in range(30)]
        references= [" " for i in range(2252)]
        transcripts=[" " for i in range(2252)]
        model=["" for i in range(2252)]
        qe =[i for i in range(2252)]
        qemean = [i for i in range(2252)]
        reader = csv.DictReader(
            f,
            dialect="excel",
            fieldnames=row,
        )
        for r in reader:
            #print(r)
            if r["row"]!="row":
                references[int(r["row"])]=r["reference transcript"]
                print(references[int(r["row"])])
                qe[int(r["row"])]=[float(r[i].split(",")[0][1:]) for i in probabilities]
                qemean[int(r["row"])]=[float(r[i].split(",")[1][:-1])for i in probabilities]
                
                currqe = [float(r[i].split(",")[1][:-1])for i in probabilities]
                transcripts[int(r["row"])]=[r[i] for i in csvtranscripts]
                model[int(r["row"])]=transcripts[int(r["row"])][currqe.index(max(currqe))]
                print(len(model),len(references))
        ref= werpy.wers( werpy.normalize(references), werpy.normalize(model))
        qe1=[math.fsum(i)/30 for i in qe]
        qe2=[math.fsum(i)/30 for i in qemean]
        qe3=[variance(i) for i in qe]
        qe4=[variance(i) for i in qemean]
        #print(ref)
        corr = pearsoncorr(qe1, ref)
        corr2 = pearsoncorr(qe2,ref)
        corr3= pearsoncorr(qe3,ref)
        corr4=pearsoncorr(qe4,ref)

        print(corr,corr2)

