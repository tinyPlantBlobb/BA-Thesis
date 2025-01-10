import re
import qeLogic
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr, worderror
import os
import csv
import math

TMPDIR = "/pfs/data5/home/kit/stud/utqma/workspaces/pfs5wor7/utqma-finals"
transcrptqes = ["transcript probability " + str(i) for i in range(0, 30)]
# TMPDIR = os.environ["TMPDIR"]
row = [
    "row",
    "reference transcript",
    "reference translation",
    "regulartranslation",
    "qe",
]
transcriptindex = ["transcript probability " + str(i) for i in range(0, 30)]
row.extend(["transcript probability " + str(i) for i in range(0, 30)])
row.extend(["transcript " + str(i) for i in range(0, 30)])
row.extend(["translation probability" + str(i) for i in range(0, 30)])
translationindex = ["translation probability" + str(i) for i in range(0, 30)]
row.extend(["translation " + str(i) for i in range(0, 30)])
with open(TMPDIR + "/translation0.csv", "r", newline="") as file:
    reader = csv.DictReader(
        file,
        dialect="excel",
        fieldnames=row,
    )
    transcripts = []
    translation = []
    reference_translation = []
    reference_trancsript = []
    transcriptionprob = []
    transcriptmean = []
    transcrptprobabiltiy = []
    transcriptmeandp = []
    transcriptcombomean=[]
    transcriptcombo=[]
    transcrptprobabiltiydp = []

    translationEntropy = []
    translationstddiv = []
    translationProbability = []
    translationqe = []
    translationdpprob = []
    translationdpvarinace = []
    translationdpcombo=[]
    translationEntropyvariance = []
    translationstddivvariance = []
    uniscore=[]
    tpscore = []
    softmaxent = []
    stddiv = []
    besttranscriptqe=[]
    for r in reader:
        # print(r)
        if r["qe"] != "qe":
            transcripts.append(r["transcript 0"])
            translation.append(r["regulartranslation"])
            reference_translation.append(r["reference translation"])
            reference_trancsript.append(r["reference transcript"])
            # print(r["qe"][1:-1].split(", "))
            qe = list(map(float, r["qe"].strip("[]()").split(",")))
            # transcriptionprob.append(r["transcript prob"])
            
            transcriptionprob.append(
                [
                    [float(j.strip()) for j in r[i].strip("([])").split(",")]
                    for i in transcriptindex
                ]
            )
            besttranscriptqe.append(max([[float(j.strip()) for j in r[i].strip("([])").split(",")]for i in transcriptindex], key=lambda x: x[1]))

            #print(len(transcriptionprob), " probability  ", len(transcriptionprob[-1]))
            transcriptmeandp.append([t[1] for t in transcriptionprob[-1]])
            transcriptmean.append(max([t[1] for t in transcriptionprob[-1]]))
            transcrptprobabiltiy.append(max([i[0] for i in transcriptionprob[-1]]))
            transcrptprobabiltiydp.append([i[0] for i in transcriptionprob[-1]])
            tpscore.append(qe[0])
            
            qes = [
                list(map(float, r[i].strip("([])").split(",")))
                for i in translationindex
            ]
            softmaxent.append(qe[1])
            stddiv.append(qe[2])
            translationqe.append(qes)
            translationdpprob.append(math.fsum([i[0] for i in qes]))
            translationdpvarinace.append(qeLogic.variance([i[0] for i in qes]))
            translationEntropy.append(math.fsum([i[1] for i in qes]))
            translationEntropyvariance.append(qeLogic.variance([i[1] for i in qes]))
            translationstddiv.append(math.fsum([i[2] for i in qes]))
            translationstddivvariance.append(qeLogic.variance([i[2] for i in qes]))

    refscores = cometscore(reference_trancsript, translation, reference_translation)
    transcriptdpmean=[sum(i)/30 for i in transcriptmeandp]
    transcriptdpprob=[sum(i)/30 for i in transcrptprobabiltiydp]
    transcriptmeandpvariance = [qeLogic.variance(i) for i in transcriptmeandp]
    transcriptprobdpvariance=[qeLogic.variance(i) for i in transcrptprobabiltiydp]
    transcriptcombomean=[1-i[0]/i[1] for i in zip(transcriptdpmean, transcriptmeandpvariance)]
    transcriptcombo=[1-i[0]/i[1] for i in zip(transcriptdpprob, transcriptprobdpvariance)]
    translationdpcombo=[1-i[0]/i[1] for i in zip(translationdpprob,translationdpvarinace)]

    refscore = refscores["scores"]
    wer = worderror(transcripts, reference_trancsript)
    refuniscores=[i[0]*(1-i[1]/max(wer)) for i in zip(refscore, wer)]
    uniscore=[i[0]*i[1] for i in zip(tpscore, transcrptprobabiltiy)]
    uniscore2=[i[0]*i[1] for i in zip(tpscore, transcriptmean)]

    with open("referencescores.txt", "w") as export:
        for i in zip(refscore,wer):
            #print(i)
            export.write(str(i[0])+"\t"+ str(i[1]))
            export.write("\n")
    #print("transcript, transcript mean, translation, softmax, stddiv, transcript dp, transcript dp var, transcript dp combo, transcirpt mean dp, transcirpt mean dp var, transcript mean dp combo, dp mean, dp variance, combo,uni1, uni2, wer, comet, uni", file=open("seamlessallscores.txt","w"))
    #print(len(transcrptprobabiltiy), "\t", len(transcriptmean),"\t",len(tpscore), "\t", len(softmaxent), "\t", len(stddiv), "\t", len(transcriptdpprob),"\t",len(transcriptprobdpvariance),"\t", len(transcriptcombo),"\t", len(transcriptdpmean), "\t",len(transcriptmeandpvariance), "\t", len(transcriptcombomean), "\t", len(uniscore), "\t", len(uniscore2),"\t", len(wer), "\t", len(refscore), "\t", len(refuniscores))
    for i in range(2255):
        print(transcrptprobabiltiy[i], "\t", transcriptmean[i],"\t",tpscore[i], "\t", softmaxent[i], "\t", stddiv[i], "\t", transcriptdpprob[i],"\t",transcriptprobdpvariance[i],"\t", transcriptcombo[i],"\t", transcriptdpmean[i], "\t",transcriptmeandpvariance[i], "\t", transcriptcombomean[i], "\t",translationdpprob[i],"\t",translationdpvarinace[i],"\t", translationdpcombo[i],"\t", uniscore[i], "\t", uniscore2[i],"\t", wer[i], "\t", refscore[i], "\t", refuniscores[i], file=open("seamlessallscores.txt","a"))
    #print(len(wer), len(transcriptmean), len(transcriptmean))
    # print(transcriptmean)
    #meanresult = pearsoncorr(transcriptmean, wer)
    # transcription results
    transcriptprobdpresult= pearsoncorr(transcriptdpprob, wer)
    transcriptmeandpres = pearsoncorr(transcriptdpmean, wer) 
    transcriptresult = pearsoncorr(transcrptprobabiltiy, wer)
    transcriptmeanres = pearsoncorr(transcriptmean, wer)
    transcriptprobvariance = pearsoncorr(transcriptprobdpvariance, wer)
    transcriptmeanvariance = pearsoncorr(transcriptmeandpvariance, wer)
    transcriptmeancombocorr = pearsoncorr(transcriptcombomean, wer)
    transcriptcombocorr = pearsoncorr(transcriptcombo, wer)

    tpresult = pearsoncorr(tpscore, refscore)
    softres = pearsoncorr(softmaxent, refscore)
    stdres = pearsoncorr(stddiv, refscore)
    unifiedcorr=pearsoncorr(uniscore, refuniscores)
    unified2corr=pearsoncorr(uniscore2, refuniscores)

    dpprob1= pearsoncorr(translationdpprob,refscore)
    dpprob2= pearsoncorr(translationdpvarinace,refscore)
    dpprobcombo=pearsoncorr(translationdpcombo, refscore)
    dpprob3= pearsoncorr(translationEntropy,refscore)
    dpprob4= pearsoncorr(translationEntropyvariance,refscore)
    dpprob5= pearsoncorr(translationstddiv,refscore)
    dpprob6= pearsoncorr(translationstddivvariance,refscore)

    #print(dpprob1)
    with open("scores.txt", "w") as resfile:
        # resfile.write("reference scores\n")
        resfile.write("transcription result \n")
        resfile.write(str(transcriptresult["pearsonr"]))
        resfile.write("\n trasncript mean ")
        resfile.write(str(transcriptmeanres["pearsonr"]))
        resfile.write("\n dropout mean variance ")
        resfile.write(str(transcriptmeanvariance["pearsonr"]))
        resfile.write("\n probability dropout variance ")
        resfile.write(str(transcriptprobvariance["pearsonr"]))
        resfile.write("\n PROBABILTIY DROPOUT ")
        resfile.write(str(transcriptprobdpresult["pearsonr"]))
        resfile.write("\n MEAN DROPOUT ")
        resfile.write(str(transcriptmeandpres["pearsonr"]))
        resfile.write("\n transcript combo")
        resfile.write(str(transcriptcombocorr["pearsonr"]))
        resfile.write("\n transcript mean combo")
        resfile.write(str(transcriptmeancombocorr["pearsonr"]))
        # resfile.write(str(refscore))
        resfile.write("\n pearsoncorr of the translation probability")
        resfile.write(str(tpresult))
        resfile.write("\nsoftmax entropy correlation\n")
        resfile.write(str(softres))
        resfile.write("\n standart div\n")
        resfile.write(str(stdres))
        resfile.write("\n dropout prob\n")
        resfile.write(str(dpprob1["pearsonr"]))
        resfile.write("\n dropout variance\n")
        resfile.write(str(dpprob2["pearsonr"]))
        resfile.write(str(dpprobcombo["pearsonr"]))
        resfile.write("\n dropout entropy mean\n")
        resfile.write(str(dpprob3["pearsonr"]))
        resfile.write("\n dropout entropy variance\n")
        resfile.write(str(dpprob4["pearsonr"]))
        resfile.write("\n dropout std mean\n")
        resfile.write(str(dpprob5["pearsonr"]))
        resfile.write("\n dropout std variance \n")
        resfile.write(str(dpprob6["pearsonr"]))
        resfile.write("\n unified score")
        resfile.write(str(unifiedcorr["pearsonr"]))
        resfile.write(" \t ")
        resfile.write(str(unified2corr["pearsonr"]))
        resfile.close()
        for a in range(100):
            uniscore=[i[0]*a/100 + (1-a/100)*i[1] for i in zip(tpscore, transcrptprobabiltiy)]
            uniscore2=[i[0]*a/100+(1-a/100)*i[1] for i in zip(tpscore,transcriptmean)]
            uniscoreb=[pow(i[0],(1-a/100))+pow(i[1],a/100) for i in  zip(tpscore, transcrptprobabiltiy)]
            uniscoreb2=[pow(i[0],(1-a/100))+pow(i[1],a/100) for i in zip(tpscore,transcriptmean)]

            print(a,"\t",pearsoncorr(uniscore, refuniscores)["pearsonr"],"\t",pearsoncorr(uniscore2, refuniscores)["pearsonr"],"\t",pearsoncorr(uniscoreb, refuniscores)["pearsonr"],"\t",pearsoncorr(uniscoreb2, refuniscores)["pearsonr"], file=open("seamlessuniscores.txt","a"))
  
