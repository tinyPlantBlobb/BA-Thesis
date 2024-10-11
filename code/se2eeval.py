# import csv
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
