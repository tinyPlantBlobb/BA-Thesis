import csv
import re
import numpy as np
import qeLogic

with open(
    "/home/plantpalfynn/Downloads/en-de/word-probas/word_probas.dev.ende", "r"
) as file:
    lines = file.readlines()
    lines = [re.findall(r"-?\d.\d+", line) for line in lines]
    lines = [[np.multiply(float(x), np.exp(float(x))) for x in line] for line in lines]
    tp = [np.sum([float(x) for x in line]) for line in lines]
    result = []
    for i in lines:
        result.append(np.divide(-np.sum(i), len(i)))
    with open(
        "/home/plantpalfynn/Downloads/en-de/dev.ende.df.short.tsv", "r"
    ) as refscores:
        reader = csv.reader(refscores, delimiter="\t")
        scores = []
        for i in reader:
            if i[0] == "index":
                continue

            scores.append(i)
        refscoreszmean = [float(row[6]) for row in scores]
        refscoresmean = [float(row[4]) for row in scores]
        refmodel = [float(row[7]) for row in scores]
        finalscore = qeLogic.pearsoncorr(result, refscoresmean)
        reference = qeLogic.pearsoncorr(refmodel, result)
        reference2 = qeLogic.pearsoncorr(refmodel, tp)
        reference3 = qeLogic.pearsoncorr(tp, result)
    print(finalscore, reference, reference2, reference3)
