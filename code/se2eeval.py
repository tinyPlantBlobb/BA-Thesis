import csv
import re
import qeLogic

#
# with open("results/seamlesse2eresult.csv", "w") as resultfile:
#    row = [
#        "row",
#        "reference transcription",
#        "reference translation",
#        "model translation",
#        "Quality estimation nondropout",
#        "",
#    ]
#    row.extend(["dropoutresult" + str(i) for i in range(30)])
#    csvwriter = csv.writer(resultfile, dialect="excel")
#
#    csvwriter.writerow(row)
#    for i in range(8):
#        with open(
#            "results/results/seamlesse2efulltranscriptions" + str(i) + ".csv", "r"
#        ) as interimf:
#            reader = csv.reader(interimf, dialect="excel")
#            for j in reader:
#                if j[0] == "row":
#                    print("aaa")
#                    continue
#                csvwriter.writerow(j)
# model_translation = "transcription"

with open("results/seamlesse2eresult.csv", "r") as resultfile:
    row = [
        "row",
        "reference transcription",
        "reference translation",
        "model translation",
        "Quality estimation nondropout",
        "empty",
    ]
    dropoutes = ["dropoutresult" + str(i) for i in range(30)]
    row.extend(dropoutes)
    csvreader = csv.DictReader(resultfile, dialect="excel", fieldnames=row)
    values = []
    for i in csvreader:
        if i["row"] == "row":
            continue
        nondopqe = [
            float(i)
            for i in re.findall(
                r"-?\d+\.\d+",
                i["Quality estimation nondropout"],
            )
        ]

        runs = [
            (
                i["reference translation"],
                i["reference transcription"],
                i["model translation"],
                nondopqe,
            )
        ]

        for j in dropoutes:
            qe = [
                float(i)
                for i in re.findall(
                    r"-?\d+\.\d+",
                    re.findall(r"\(-?\d+\.\d*, -?\d+\.\d*, -?\d+\.\d*\)\)\"?,?", i[j])[
                        0
                    ],
                )
            ]
            runs.append((i[j].split(", (")[0], qe[0], qe[1], qe[2]))

            # runs.append((list(map(float, spit[1][:-1].split(", "))), spit[0][1:]))

            # print(runs[-1])
        # print(runs[1], type(runs[1][2]))
        qeestimate = qeLogic.calcdropoutprob(runs)
        qevariance = qeLogic.calcdropoutvariance(runs)
        # print(qeestimate)
        values.append((runs, qeestimate, qevariance, (1 - (qeestimate / qevariance))))
        # print(qeestimate)
    # print(values[0][0][3])
    # first element of of values tuple=runs, first elements of runs = non dropout data tuple = ref transcription, ref translation, model translation (tpqe, softmax, diviation)
    # TODO add lexsim stuff here

    referencetranscrits = [i[0][0][0] for i in values]
    referencetransaltion = [i[0][0][1] for i in values]
    modeltransaltion = [i[0][0][2] for i in values]

    # regular translation prob qe that is normalised
    # print(values[0][0][3], values[0][0][3][3])
    refentropyqe = [i[0][0][3][1] for i in values]
    refstddivqe = [i[0][0][3][2] for i in values]
    referenceqe = [i[0][0][3][0] for i in values]
    # print(reftranslationqe[0], referenceqe[0])
    dropoutcalculatedqe = [i[1] for i in values]

    refscores = qeLogic.cometscore(
        referencetranscrits,
        modeltransaltion,
        referencetransaltion,
    )

    qescores = qeLogic.pearsoncorr(dropoutcalculatedqe, refscores["scores"])
    qevariancecorr = qeLogic.pearsoncorr([i[2] for i in values], refscores["scores"])
    qecombocorr = qeLogic.pearsoncorr([i[3] for i in values], refscores["scores"])
    qescore2 = qeLogic.pearsoncorr(referenceqe, refscores["scores"])
    qescore4 = qeLogic.pearsoncorr(refstddivqe, refscores["scores"])
    qescore3 = qeLogic.pearsoncorr(refentropyqe, refscores["scores"])
    print(
        "dropout tp corr",
        qescores["pearsonr"],
        "variance corr",
        qevariancecorr["pearsonr"],
        "combo corr",
        qecombocorr["pearsonr"],
        "corr with mean tp qe",
        qescore2["pearsonr"],
        "corr with entropy qe",
        qescore3["pearsonr"],
        "corr with ref std div tp ",
        qescore4["pearsonr"],
    )
