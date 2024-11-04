import re
import csv
import qeLogic
import numpy as np

with open("results/dlmresults.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        [
            "Target",
            "Probabilities",
            # "Hypothesis",
            "Source Sentences",
            "Detokenized Hypothesis",
            "qe",
        ]
    )
    dptpqe = [0.0] * 3000
    dpvar = [0.0] * 3000
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
            row = []
            for j in range(len(detokenizedhypothesis) - 1):
                # print(i)
                dptpqe[j] += float(detokenizedhypothesis[j][0])

                row.extend(
                    [
                        target[j],
                        probabilities[j],
                        # hypothesis[i],
                        srcsentences[j],
                        detokenizedhypothesis[j][1],
                        detokenizedhypothesis[j][0],
                    ]
                )

            csvwriter.writerow(row)
            # print(target[i])
            # print(detokenizedhypothesis)
    dpvar = qeLogic.variance(dptpqe).cpu().numpy()
    # dpcombo = [(1.0 - ((i[0] / 30.0) / i[1])) for i in zip(dptpqe, dpvar) if i[1] != 0]
    # print(dpcombo)
    print(dptpqe, dpvar)
