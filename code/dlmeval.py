import re
import csv
import qeLogic
import numpy as np
import torch
from itertools import groupby

BASE = "/pfs/work7/workspace/scratch/utqma-finals/results-25030350/"
BASE = "results/results-25078023/"

ref = []
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
    softmaxentropy = []
    variances = []
    probs = []
    with open(BASE + "generate-test.txt", "r") as f:
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

        softmaxentropy = [-sum(i) / len(i) for i in vocabscores][:2255]
        variances = [qeLogic.stddiv(i) for i in probabilities][:2255]
        probs = [i[0] for i in detokenizedhypothesis][:2255]

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
            ]
            for i in range(len(target))
        ]

        row.extend(line)
        # csvwriter.writerows(row)
        hypothesis = [i[1] for i in detokenizedhypothesis]
        # print(len(hypothesis), len(target), len(srcsentences))
        ref = qeLogic.comet(srcsentences, hypothesis, target)["scores"]

    probabilitycorrelation = qeLogic.pearsoncorr(probs, ref)
    softmaxcorrelation = qeLogic.pearsoncorr(softmaxentropy, ref)
    varcorrelation = qeLogic.pearsoncorr(variances, ref)
    print(
        "probabilitycorrelation",
        probabilitycorrelation,
        "softmaxcorrelation",
        softmaxcorrelation,
        "varcorrelation",
        varcorrelation,
    )
    # for i in range(1, 30):
    with open(BASE + "sorted.txt", "r") as f:
        lines = f.readlines()
        print(len(lines))
        output = [i.strip() for i in lines if re.search(r"^[VTPHSD]-\d.*", i)]

        output.sort(key=lambda x: int(re.search(r"^[TPDHSV]-(\d+)", x).group(1)) % 2255)
        # keys=[]
        # groups = []
        # alllinesdict={1:[]}

        # for key, group in groupby(output, lambda x:(int(re.search(r"^[TDPHSV]-(\d+)",x).group(1))%2252)):
        #    if key:
        #        keys.append(key)
        #        groups.append(list(group))
        #        dictelem = alllinesdict.get(key,[])
        #        print(key,dictelem)
        #        alllinesdict[key] =dictelem.append(list(group))
        #
        # print(list(groups), file=open("tst","w"))
        #        #print
        #     print([i.split("\t")[0]for i in output], file=open("out.txt","w"))
        target = [
            j.split("\t")[1] if len(j.split("\t")) > 1 else " "
            for j in output
            if re.search(r"^T-\d+.*", j)
        ]
        #     #print([i.split("\t") for i in re.findall(r"^T-\d+.*", i) ])
        test = [i.split("\t")[0] for i in output if re.search(r"^T-\d+", i)]
        #
        probabilities = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"^P-\d.*", i)
        ]
        hypothesis = [
            i.split("\t")[1:]
            for i in output
            if re.search(r"^H-\d.*", i) and len(i.split("\t")) > 1
        ]
        srcsentences = [
            i.split("\t")[1:] if len(i.split("\t")) > 1 else " "
            for i in output
            if re.search(r"^S-\d.*", i)
        ]
        detokenizedhypothesis = [
            i.split("\t")[1:]
            for i in output
            if re.search(r"^D-\d.*", i) and len(i.split("\t")) > 1
        ]
        vocabscores = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"V-\d.*", i)
        ]
        # probabilities.sort(key=lambda x: int(re.search(r"^[TPHDSV]-(\d+)",x)))
        # hypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        # srcsentences.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        # detokenizedhypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        # target.sort(key=lambda x:int(re.search(r"^[TDPHSV]-(\d+)",x)))

        for i in range(len(row)):
            # print(j)
            for j in range(30):
                # print(i,j,i*j, i*j+j, detokenizedhypothesis[i*j], file=open("tst","a"))
                dptpqe[i] += float(detokenizedhypothesis[30 * i + j][0])
                # row[i][5].append(target[j])
                row[i][6].append(detokenizedhypothesis[30 * i + j][1])
                row[i][8].append(float(detokenizedhypothesis[30 * i + j][0]))
                print(row[i][8], detokenizedhypothesis[30 * i + j])
            # print(len(target), len(row), )
            # print(i, j, target[j*i])
            row[i][7].extend(probabilities[i * j : i * j + 30])
            # row[i][7].append(srcsentences[j])
            # row[j + 1][10].append(-sum(vocabscores[j*i]) / len(vocabscores[j*i]))
        # [print(row[i][8]) for i in range(2252)]
        # lexsims = [qeLogic.lexsim(i[6]) for i in row]
        csvwriter.writerows(row)
    dpvar = [qeLogic.variance(row[i][8]).cpu().numpy().item() for i in range(2255)]
    dptqe = [i / 30 for i in dptpqe]
    # print(len(ref), len(probs), dptpqe)
    combo = [1 - dptqe[i] / dpvar[i] for i in range(len(dpvar))]
    probabilitycorrelation = qeLogic.pearsoncorr(probs, ref)
    softmaxcorrelation = qeLogic.pearsoncorr(softmaxentropy, ref)
    varcorrelation = qeLogic.pearsoncorr(variances, ref)
    dpprob = qeLogic.pearsoncorr(dptqe, ref)
    dpvarcorr = qeLogic.pearsoncorr(dpvar, ref)
    dpcombo = qeLogic.pearsoncorr(combo, ref)
    # lexsimres = qeLogic.pearsoncorr(lexsims, ref)
    print(
        "probabilitycorrelation",
        probabilitycorrelation,
        "softmaxcorrelation",
        softmaxcorrelation,
        "std correlation",
        varcorrelation,
        "dp prob corr",
        dpprob,
        "dp var corr",
        dpvarcorr,
        "dpcombo",
        dpcombo,
        "lexsim",
        # lexsimres,
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
            ref[i],
            file=open("dlmprint.txt", "a"),
        )
