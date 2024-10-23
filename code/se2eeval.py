import csv
from os import walk
import re
import qeLogic

#
# with open("results/seamlessfulltransctiptions.csv", "w") as resultfile:
#    row = [
#        "row",
#        "reference transcription",
#        "reference translation",
#        "Quality estimation nondropout",
#        "",
#    ]
#    row.extend(["dropoutresult" + str(i) for i in range(30)])
#    csvwriter = csv.writer(resultfile, dialect="excel")
#
#    csvwriter.writerow(row)
#    for i in range(8):
#        with open(
#            "results/seamlessfulltranscriptions" + str(i) + ".csv", "r"
#        ) as interimf:
#            reader = csv.reader(interimf, dialect="excel")
#            for j in reader:
#                if j[0] == "row":
#                    print("aaa")
#                    continue
#                csvwriter.writerow(j)
with open("results/seamlesse2eresults.csv", "r") as resultfile:
    row = [
        "row",
        "reference transcription",
        "reference translation",
        "Quality estimation nondropout",
        "empty",
    ]
    dropoutes = ["dropoutresult" + str(i) for i in range(30)]
    row.extend(dropoutes)
    csvreader = csv.DictReader(resultfile, dialect="excel")
    values = []
    for i in csvreader:
        runs = [
            (
                i["reference transcription"],
                i["reference translation"],
                i["Quality estimation nondropout"],
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

        values.append((runs, qeestimate))
        print(qeestimate)

    referencetranscrits = [i[0][0] for i in values]
    referencetransaltion = [i[0][1] for i in values]
    refqe = [i[0][2] for i in values]
    refscores = cometscore(
        referencetranscrits, referencetransaltion, reference_translation
    )
