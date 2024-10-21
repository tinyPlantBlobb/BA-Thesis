import csv

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
            spit = i[j].split(", (")

            runs.append((list(map(float, spit[1][:-2].split(", "))), spit[0][1:]))

            print(runs[-1])
        values.append(runs)
