import re
from datasets.features import translation
from qeLogic import pearsoncorr, worderror
import os
import csv

TMPDIR = "/home/plantpalfynn/uni/BA/BA-Thesis/code/"
transcripts = []

translation = []
reference_translation = []
reference_trancsript = []
transcriptionprob = []
transcriptmean = []
tpscore = []
softmaxent = []
stddiv = []
fullthing = []
row = ["row", "reference transcript", "reference translation", "qe"]
qes = ["transcript probability " + str(i) for i in range(0, 30)]
transcriptindex = ["transcript probability " + str(i) for i in range(0, 30)]
row.extend(["transcript probability " + str(i) for i in range(0, 30)])
row.extend(["transcript " + str(i) for i in range(0, 30)])
for i in range(4):
    with open(
        TMPDIR
        + "results/results-24260948/dropouttranslationfulldropped"
        + str(i)
        + ".csv",
        "r",
    ) as resfile:
        reader = csv.DictReader(
            resfile,
            dialect="excel",
            fieldnames=row,
        )
        for r in reader:
            if r["row"] != "row":
                # print(type(r))
                fullthing.append(r)


with open("seamlessdropout.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=row)
    writer.writerows(sorted(fullthing, key=lambda x: int(x["row"])))
reference_scores = []
with open("referencescores.txt", "r") as ref:
    lines = ref.readlines()
    reference_scores = [float(i) for i in lines]
with open(("seamlessdropout.csv"), "r") as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=row)
    for row in reader:
        # transcripts.append(r["transcription"])
        translation.extend([row[i] for i in transcriptindex])
        reference_translation.append(row["reference translation"])
        reference_trancsript.append(row["reference transcript"])
        # print(r["qe"][1:-1].split(", "))
        # qe = r["qe"][1:-1].split(", ")
        transcriptionprob.append(
            [[float(j) for j in re.findall(r"-?\d+\.\d*", row[i])] for i in qes]
        )
    print(len(transcriptionprob))
    cutoff = len(transcriptionprob)
    translationProbability = [sum([j[0] for j in i]) / 30 for i in transcriptionprob]

    # softmaxent = [i[1] for i in transcriptionprob]
    # stddiv = [i[2] for i in transcriptionprob]
    result = pearsoncorr(translationProbability, reference_scores[:cutoff])
    # result2 = pearsoncorr(softmaxent, reference_scores[:cutoff])
    # result3 = pearsoncorr(stddiv, reference_scores[:cutoff])
    print(result)
