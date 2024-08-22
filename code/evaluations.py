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
    trans = translation = reference = tpscore = softmaxent = stddiv = []

    for r in reader:
        trans.append(r["transcript"])
        translation.append(r["translation"])
        reference.append(r["reference"])
        print(r["qe"])
        qe = r["qe"]
        if qe != "qe":
            tpscore = r["qe"][0]
            softmaxent = r["qe"][1]
            stddiv = r["qe"][2]
    refscore = cometscore(trans, translation, reference)

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
