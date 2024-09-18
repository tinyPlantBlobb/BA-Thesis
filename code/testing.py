from os import walk
from datasets.features import translation
import torch
from qeLogic import TranslationProbability, softmaxEntropy, sentStd, readCSV
from datasets import Dataset, load_dataset

BASE = "/home/plantpalfynn/uni/BA/BA-Thesis/code/"
# dataset1 = load_dataset(
#    "csv", data_files=BASE + "results/dropoutfulltranscriptions.csv"
# )
dataset = Dataset.from_dict(readCSV(BASE + "results/dropoutfulltranscriptions.csv"))
print(dataset)


# with open(
#    "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/seamless_result0.pt", "rb"
# ) as f:
#    print(f)
#    data = torch.load(
#        "/home/plantpalfynn/uni/BA/BA-Thesis/code/results/seamless_result0.pt",
#        map_location=torch.device("cuda"),
#    )
#
#    # Load data from file
#    # print(type(data[0]))
#    generationdata = data[0]
#    qeres = getQE(generationdata, translation=True, dropout=False)
