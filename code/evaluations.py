import re
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr
import os
import csv

TMPDIR = os.environ["TMPDIR"]

with open(TMPDIR + "/results/seamlessfulltranscriptions.csv", "r", newline="") as file:
    reader = csv.DictReader(
        file,
        dialect="excel",
        fieldnames=["row", "transcript", "reference", "translation", "qe"],
    )
    trans = []
    translation = []
    reference = []
    tpscore = []
    softmaxent = []
    stddiv = []

    for r in reader:
        trans.append(r["transcript"])
        translation.append(r["translation"])
        reference.append(r["reference"])
        print(r["qe"])
        qe = r["qe"]
        if qe != "qe":
            tpscore.append(qe[0])
            softmaxent.append(qe[1])
            stddiv.append(qe[2])
    refscores = cometscore(trans, translation, reference)
    refscore = refscores["scores"]
    with open(TMPDIR + "/resultscore.csv", "w") as resscorefile:
        reswriter = csv.writer(resscorefile, dialect="excel")
        reswriter.writerow(
            [
                "row",
                "transcript",
                "reference",
                "translation",
                "qe tp",
                "qe softent",
                "qe stddiv",
                "refscore",
            ]
        )
        for i in range(len(translation)):
            reswriter.writerow(
                [
                    i,
                    trans[i],
                    reference[i],
                    translation[i],
                    tpscore[i],
                    softmaxent[i],
                    stddiv[i],
                    refscore[i],
                ]
            )
        resscorefile.close()
    print(type(refscore))
    print(len(trans), len(translation), len(reference))
    tpresult = pearsoncorr(tpscore, refscore)
    softres = pearsoncorr(softmaxent, refscore)
    stdres = pearsoncorr(stddiv, refscore)
    with open(TMPDIR + "/results/scores.txt", "w") as resfile:
        resfile.write("reference scores\n")
        resfile.write(refscore)
        resfile.write("\n\n\n pearsoncorrelation of the Translation Probability")
        resfile.write(tpresult)
        resfile.write("\nsoftmax correlation\n")
        resfile.write(softres)
        resfile.write("\n standart div\n")
        resfile.write(stdres)
        resfile.close()
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
