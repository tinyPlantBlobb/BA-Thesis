from qeLogic import cometscore, pearsoncorr
import os
import csv

TMPDIR = os.environ["TMPDIR"]

with open(TMPDIR + "/seamless_result.csv", "r", newline="") as file:
    reader = csv.DictReader(
        file,
        dialect="excel",
        fieldnames=["row", "transcript", "reference", "translation", "qe"],
    )
    refscore = [
        cometscore(row["transcript"], row["translation"], row["reference"])
        for row in reader
    ]
    tpscore = [row["qe"][0] for row in reader]

    result = pearsoncorr(tpscore, refscore)
    print(result)
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
