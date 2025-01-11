import re
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr, wer as worderror
import os
import csv

TMPDIR = os.environ["TMPDIR"]
filepath=TMPDIR + "/results/seamlessfulltranscriptions.csv" 
#filepath = ""
with open(filepath, "r", newline="") as file:
    reader = csv.DictReader(
        file,
        dialect="excel",
        fieldnames=[
            "row",
            "reference transcript",
            "reference translation",
            "transcription",
            "translation",
            "transcript prob",
            "transcript mean",
            "qe",
        ],
    )
    transcripts = []
    translation = []
    reference_translation = []
    reference_trancsript = []
    transcriptionprob = []
    transcriptmean = []
    tpscore = []
    softmaxent = []
    stddiv = []
    fullscore=[]

    for r in reader:
        # print(r)
        if r["qe"] != "qe":
            transcripts.append(r["transcription"])
            translation.append(r["translation"])
            reference_translation.append(r["reference translation"])
            reference_trancsript.append(r["reference transcript"])
            # print(r["qe"][1:-1].split(", "))
            qe = r["qe"][1:-1].split(", ")
            transcriptionprob.append(float(r["transcript prob"]))
            print(r["transcript mean"], " probability  ", r["transcript prob"])
            transcriptmean.append(float(r["transcript mean"]))
            tpscore.append(float(qe[0]))
            softmaxent.append(float(qe[1]))
            stddiv.append(float(qe[2]))
            fullscore.append(float(qe[0])*float(r["transcript mean"]))
    refscores = cometscore(transcripts, translation, reference_translation)

    refscore = refscores["scores"]
    with open(TMPDIR + "/results/referencescores.txt", "w") as export:
        for i in refscore:
            export.write(str(i))
            export.write("\n")
    wer = [
        worderror(transcripts, reference_trancsript)
        for i in range(len(transcripts))
    ]
    # print(wer, len(wer), len(transcriptmean))
    # print(transcriptmean)
    meanresult = pearsoncorr(transcriptmean, wer)
    tpresult = pearsoncorr(tpscore, refscore)
    softres = pearsoncorr(softmaxent, refscore)
    stdres = pearsoncorr(stddiv, refscore)
    transcriptresult = pearsoncorr(transcriptionprob, wer)
    reffullscore = [i[0]/i[1] for i in zip(cometscore, wer)]
    fullscorecor= pearsoncorr(fullscore, reffullscore)
    print(transcriptresult, tpresult, softres, stdres)
    with open(TMPDIR + "/results/scores.txt", "w") as resfile:
        # resfile.write("reference scores\n")
        # resfile.write(str(refscore))
        resfile.write("\n\n\n pearsoncorr of the translation probability")
        resfile.write(str(tpresult["pearsonr"]))
        resfile.write("\nsoftmax correlation\n")
        resfile.write(str(softres["pearsonr"]))
        resfile.write("\n standart div\n")
        resfile.write(str(stdres["pearsonr"]))
        resfile.write("\n transcription result \n")
        resfile.write(str(transcriptresult["pearsonr"]))
        resfile.write("\n trasncript mean\n")
        resfile.write(str(meanresult["pearsonr"]))
        resfile.write("\n fullscore\n")
        resfile.write(str(fullscorecor["pearsonr"]))
        resfile.close()
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
