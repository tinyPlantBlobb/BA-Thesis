import csv

with open("deltalmfull.csv", "r") as file:
    row = [
        "Target",
        "Probabilities",
        "Source Sentences",
        "Detokenized Hypothesis",
        "Vocabulary scores",
        "qe",
    ]
    reader = csv.DictReader(file, fieldnames=row)
    for line in reader:
        line[row[3]].split()
