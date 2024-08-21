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
    trans = translation = reference = tpscore = []
    for r in reader:
        trans.append(r["transcript"])
        translation.append(r["translation"])
        reference.append(r["reference"])
        tpscore = r["qe"][0]
    refscore = [cometscore(trans, translation, reference) for row in reader]

    print(len(trans), len(translation), len(reference))
    result = pearsoncorr(tpscore, refscore)
    print(result)
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
