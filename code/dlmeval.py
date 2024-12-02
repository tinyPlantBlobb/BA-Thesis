import re
import csv
import qeLogic
import numpy as np

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
    dptpqe = [0.0] * 3000
    dpvar = [0.0] * 3000
    row = [[]]
    with open("results/generate-test.txt", "r") as f:
        lines = f.readlines()
        output = [i.strip() for i in lines if re.search(r"^[TPHSDV]-\d.*", i)]
        output.sort()
        target = [i.split("\t")[1] for i in output if re.search(r"^T-\d.*", i)]
        dptpqe = dptpqe[: len(target)]
        dpvar = dpvar[: len(target)]
        probabilities = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"^P-\d.*", i)
        ]
        hypothesis = [i.split("\t")[1:] for i in output if re.search(r"^H-\d.*", i)]
        srcsentences = [i.split("\t")[1:] for i in output if re.search(r"^S-\d.*", i)]
        detokenizedhypothesis = [
            i.split("\t")[1:] for i in output if re.search(r"D-\d.*", i)
        ]
        vocabscores = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"V-\d.*", i)
        ]

        line = [
            [
                target[i],
                srcsentences[i],
                detokenizedhypothesis[i][0],
                (-sum(vocabscores[i]) / len(vocabscores[i])),
                qeLogic.variance(probabilities[i]).cpu().numpy(),
                [target[i]],
                [probabilities[i]],
                [srcsentences[i]],
                [detokenizedhypothesis[i][0]],
                [detokenizedhypothesis[i][1]],
                [(-sum(vocabscores[i]) / len(vocabscores[i]))],
                sum(probabilities[i][1:]) / len(probabilities[i][1:]),
                qeLogic.variance(probabilities[i]),
            ]
            for i in range(len(target))
        ]

        row.extend(line)

    for i in range(1, 30):
        with open("results/generate-test" + str(i) + ".txt", "r") as f:
            lines = f.readlines()

            output = [i.strip() for i in lines if re.search(r"^[TPHSD]-\d.*", i)]
            output.sort()
            target = [i.split("\t")[1] for i in output if re.search(r"^T-\d.*", i)]
            probabilities = [
                list(map(float, i.split("\t")[1].split(" ")))
                for i in output
                if re.search(r"^P-\d.*", i)
            ]
            hypothesis = [i.split("\t")[1:] for i in output if re.search(r"^H-\d.*", i)]
            srcsentences = [
                i.split("\t")[1:] for i in output if re.search(r"^S-\d.*", i)
            ]
            detokenizedhypothesis = [
                i.split("\t")[1:] for i in output if re.search(r"D-\d.*", i)
            ]
            vocabscores = [
                list(map(float, i.split("\t")[1].split(" ")))
                for i in output
                if re.search(r"V-\d.*", i)
            ]

            for j in range(len(detokenizedhypothesis) - 1):
                # print(i)
                dptpqe[j] += float(detokenizedhypothesis[j][0])
                row[j + 1][5].append(target[j])
                row[j + 1][6].append(probabilities[j])
                row[j + 1][7].append(srcsentences[j])
                row[j + 1][8].append(detokenizedhypothesis[j][1])
                row[j + 1][9].append(detokenizedhypothesis[j][0])
                row[j + 1][10].append(-sum(vocabscores[j]) / len(vocabscores[j]))
    csvwriter.writerow(
        [
            "tokentaget",
            "tokenprobabilities",
            "srcsentence",
            "detokentised hypo score",
            "detokenizedhypothesis",
            "tokenvocabscores",
        ]
    )
    csvwriter.writerows(row)
    # print(target[i])
    # print(detokenizedhypothesis)
    dpvar = qeLogic.variance(dptpqe).cpu().numpy()
    # dpcombo = [(1.0 - ((i[0] / 30.0) / i[1])) for i in zip(dptpqe, dpvar) if i[1] != 0]
    # print(dpcombo)
    print(dptpqe, dpvar)
