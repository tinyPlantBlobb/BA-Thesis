import re
import csv
import qeLogic
import numpy as np
BASE = "/pfs/work7/workspace/scratch/utqma-finals/results-25020477/"
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
    row = []
    with open(BASE+"generate-test.txt", "r") as f:
        lines = f.readlines()
        output = [i.strip() for i in lines if re.search(r"^[TPHSDV]-\d.*", i)]
        output.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x).group(1)))
        
        target = [i.split("\t")[1] for i in output if re.search(r"^T-\d.*", i) and len(i.split("\t"))>1]
        dptpqe = dptpqe[: len(target)]
        dpvar = dpvar[: len(target)]
        probabilities = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"^P-\d.*", i) and len(i.split("\t"))>1
        ]
        hypothesis = [i.split("\t")[1:] for i in output if re.search(r"^H-\d.*", i) and len(i.split("\t"))>1]
        srcsentences = [i.split("\t")[1:] for i in output if re.search(r"^S-\d.*", i) and len(i.split("\t"))>1]
        detokenizedhypothesis = [
            i.split("\t")[1:] for i in output if re.search(r"^D-\d.*", i)
        ]
        #print(detokenizedhypothesis)
        vocabscores = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"V-\d.*", i)
        ]
        
        with open("dlmprint.txt","w") as pri:
            pri.write("\n".join([k[1] for k in detokenizedhypothesis]))

        line = [[
                detokenizedhypothesis[i][1],
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

    #for i in range(1, 30):
    with open(BASE+"dropout-test.txt", "r") as f:
        lines = f.readlines()
        print(len(lines))
        output = [i.strip() for i in lines if re.search(r"^[VTPHSD]-\d.*", i)]

        output.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x).group(1)))
        print([i.split("\t")[0]for i in output], file=open("out.txt","w"))
        target = [j.split("\t")[1] if len(j.split("\t"))>1 else " " for i in output for j in re.findall(r"^T-\d+.*", i)]
        print([i.split("\t") for i in re.findall(r"^T-\d+.*", output) ])

        
        probabilities = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"^P-\d.*", i)
        ]
        hypothesis = [i.split("\t")[1:] for i in output if re.search(r"^H-\d.*", i) and len(i.split("\t"))>1]
        srcsentences = [
            i.split("\t")[1:] if len(i.split("\t"))>1 else " " for i in output if re.search(r"^S-\d.*", i)
        ]
        detokenizedhypothesis = [
            i.split("\t")[1:] for i in output if re.search(r"^D-\d.*", i) and len(i.split("\t"))>1
        ]
        vocabscores = [
            list(map(float, i.split("\t")[1].split(" ")))
            for i in output
            if re.search(r"V-\d.*", i)
        ]
        #probabilities.sort(key=lambda x: int(re.search(r"^[TPHDSV]-(\d+)",x)))
        #hypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        #srcsentences.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        #detokenizedhypothesis.sort(key=lambda x:int(re.search(r"^[TPDHSV]-(\d+)",x)))
        #target.sort(key=lambda x:int(re.search(r"^[TDPHSV]-(\d+)",x)))


        for j in range(2251*31):
            i = j % len(row)
            if(i ==0): print(len(row),len(target),i,j, target[j], "|", row[i][0])
            #print(j, len(row),len(vocabscores), len(row[j+1]), row[j+1])
            dptpqe[i] += float(detokenizedhypothesis[j][0])
            row[i][5].append(target[j])
            #print(len(target), len(row), )
            #print(i, j, target[j*i])
            row[i][6].append(probabilities[j])
            row[i][7].append(srcsentences[j])
            row[i][8].append(detokenizedhypothesis[j][1])
            row[i][9].append(detokenizedhypothesis[j][0])
        #    #row[j + 1][10].append(-sum(vocabscores[j*i]) / len(vocabscores[j*i]))
    #csvwriter.writerow(
    #    [
    #        "tokentaget",
    #        "tokenprobabilities",
    #        "srcsentence",
    #        "detokentised hypo score",
    #        "detokenizedhypothesis",
    #        "tokenvocabscores",
    #    ]
    #)
    csvwriter.writerows(row)
    # print(target[i])
    # print(detokenizedhypothesis)
    dpvar = qeLogic.variance(dptpqe).cpu().numpy()
    # dpcombo = [(1.0 - ((i[0] / 30.0) / i[1])) for i in zip(dptpqe, dpvar) if i[1] != 0]
    # #print(dpcombo)
    #print(dptpqe, dpvar)
