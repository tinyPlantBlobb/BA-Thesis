import re
import csv

with open("results/generate-test1.txt", "r") as f:
    with open("results/generate-test.csv", "w") as csvfile:
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
        srcsentences = [i.split("\t")[1:] for i in output if re.search(r"^S-\d.*", i)]
        detokenizedhypothesis = [
            i.split("\t")[1:] for i in output if re.search(r"D-\d.*", i)
        ]
        for i in range(len(detokenizedhypothesis) - 1):
            print(i)
            csvwriter.writerow(
                [
                    target[i],
                    probabilities[i],
                    # hypothesis[i],
                    srcsentences[i],
                    detokenizedhypothesis[i][1],
                    detokenizedhypothesis[i][0],
                ]
            )
            print(target[i])
        # print(detokenizedhypothesis)
# regex: [TPHSD]-\d.*
