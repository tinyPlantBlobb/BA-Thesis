import csv

with open("results/dlmresults.csv", "r") as f:
    row = [
        "Target",
        "Probabilites",
        "Source sentence",
        "Detokenized sentence",
        "Vocabulary scores",
        "qe",
    ]
    reader = csv.DictReader(f, fieldnames=row)
    data = list(reader)
