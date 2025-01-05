import matplotlib.pyplot as plt
import argparse
import csv

base = "/home/plantpalfynn/uni/BA/BA-Thesis/Latex/sections/images/"
args = argparse.ArgumentParser()
args.add_argument("--inp", type=str, default="inp.txt")
args.add_argument("--ref", type=str, default="ref.txt")
args.add_argument("--split", type=bool, default=False)
args.add_argument("--model", type=str, default="out")
ref = args.parse_args().ref
input = args.parse_args().inp
splitting = args.parse_args().split
model = args.parse_args().model
transcriptbasescore = []
transcriptmeanbasescore = []
with open("results/alltranscriptions.csv", "r") as transcriptfile:
    reader = csv.DictReader(transcriptfile, dialect="excel")
    for row in reader:
        print(row)
        transcriptbasescore.append(float(row["transcript prob"]))
        transcriptmeanbasescore.append(float(row["transcript mean"]))
with open("results/seamlessallscores.txt", "r") as reffile:
    lines = reffile.readlines()
    refs = [
        [float(j.strip("qwertzuiopasdfghjklyxcvbn() ")) for j in i.split("\t")]
        for i in lines
    ]
    # refs = list(map(lambda x: list(map(lambda: float(y), x.split("\t"))) , lines))
    model = "seamless"

    uniscore = [i[-5] for i in refs]
    uniscore2 = [i[-4] for i in refs]
    wers = [i[-3] for i in refs]
    comets = [i[-2] for i in refs]
    uniref = [i[-1] for i in refs]
    transcriptmean = [i[1] for i in refs]
    transcriptprob = [i[0] for i in refs]
    translationprob = [i[2] for i in refs]
    softmaxprob = [i[3] for i in refs]
    stddiv = [i[4] for i in refs]
    dropouttranscript = [i[5] for i in refs]
    dropouttranscriptvar = [i[6] for i in refs]
    dropouttranscriptcombo = [i[7] for i in refs]
    dropouttranscriptmean = [i[8] for i in refs]
    dropouttranscriptmeanvar = [i[9] for i in refs]
    dropouttranscriptmeancombo = [i[10] for i in refs]
    dropouttranslation = [i[11] for i in refs]
    dropouttranslationvar = [i[12] for i in refs]
    dropouttranslationcombo = [i[13] for i in refs]

    plt.scatter(wers, transcriptbasescore, color="blue", label="Probability")
    plt.ylabel("transcript Probability")
    plt.xlabel("reference score")
    plt.savefig(base + "transcriptbasescore.png")
    plt.clf()

    plt.scatter(wers, transcriptmeanbasescore, color="blue", label="Probability")
    plt.ylabel("transcript mean Probability")
    plt.xlabel("reference score")
    plt.savefig(base + "transcriptmeanbasescore.png")
    plt.clf()

    plt.scatter(comets, translationprob, color="blue", label="Probability")
    plt.ylabel("translation Probability")
    plt.xlabel("reference score")
    plt.savefig(base + model + "genprob.png")
    plt.clf()

    plt.scatter(comets, softmaxprob, color="blue", label="Entropy")
    plt.ylabel("softmax entropy")
    plt.xlabel("reference score")
    plt.savefig(base + model + "entropy.png")
    plt.clf()

    plt.scatter(comets, stddiv, color="blue", label="Standard Deviation")
    plt.ylabel("softmax standard deviation")
    plt.xlabel("reference score")
    plt.savefig(base + model + "stddiv.png")
    plt.clf()

    plt.scatter(comets, dropouttranslation, color="blue", label="Dropout Probability")
    plt.ylabel("dropout translation Probability")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropprob.png")
    plt.clf()

    plt.scatter(comets, dropouttranslationvar, color="blue", label="Dropout Variance")
    plt.ylabel("dropout variance")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropoutvariance.png")
    plt.clf()

    plt.scatter(comets, dropouttranslationcombo, color="blue", label="Dropout Combo")
    plt.ylabel("dropout combo")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropoutcombo.png")
    plt.clf()

    plt.scatter(wers, dropouttranscript, color="blue", label="Dropout Combo")
    plt.ylabel("dropout transcript Probability")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscript.png")
    plt.clf()

    plt.scatter(wers, dropouttranscriptvar, color="blue", label="Dropout Combo")
    plt.ylabel("dropout transcript variance")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscriptvar.png")
    plt.clf()

    plt.scatter(wers, dropouttranscriptcombo, color="blue", label="Dropout Combo")
    plt.ylabel("dropout transcript combo")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscriptcombo.png")
    plt.clf()

    plt.scatter(
        wers, dropouttranscriptmean, color="blue", label="Dropout transcript mean"
    )
    plt.ylabel("dropout transcript mean")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscriptmean.png")
    plt.clf()

    plt.scatter(
        wers,
        dropouttranscriptmeanvar,
        color="blue",
        label="Dropout transcript mean variance",
    )
    plt.ylabel("dropout transcript mean variance")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscriptmeanvar.png")
    plt.clf()

    plt.scatter(
        wers,
        dropouttranscriptmeancombo,
        color="blue",
        label="Dropout transcript mean combo",
    )
    plt.ylabel("dropout transcript mean combo")
    plt.xlabel("reference score")
    plt.savefig(base + model + "dropouttranscriptmeancombo.png")
    plt.clf()

    plt.scatter(uniref, uniscore, color="blue", label="Uniscore")
    plt.ylabel("uniscore")
    plt.xlabel("reference score")
    plt.savefig(base + model + "uniscore.png")
    plt.clf()

    plt.scatter(uniref, uniscore2, color="blue", label="Uniscore2")
    plt.ylabel("uniscore with mean")
    plt.xlabel("reference score")
    plt.savefig(base + model + "uniscore2.png")
    plt.clf()

    x = range(len(refs))
    # print(len(x), len(refs))
    plt.scatter(x, wers, color="blue", label="Reference")
    plt.savefig(base + model + "werref.png")
    plt.clf()
    plt.scatter(x, comets, color="blue", label="Comet")
    plt.savefig(base + model + "references.png")
    plt.clf()
if input != "ref.txt":
    with open("results/dlmallscores.txt", "r") as inpfile:
        if splitting:
            model = "dlm"
            lines = inpfile.readlines()
            cases = [l.split(",") for l in lines]
            inps = [[float(j) for j in i.split("\t")] for i in lines]
            wers = [i[-2] for i in inps]
            comets = [i[-3] for i in inps]
            uniref = [i[-1] for i in inps]

            genprop = [i[0] for i in inps]
            plt.scatter(comets, genprop, color="blue", label="Probability")
            plt.ylabel("translation Probability")
            plt.xlabel("reference score")
            plt.savefig(base + model + "genprob.png")
            plt.clf()

            entropy = [i[1] for i in inps]
            plt.scatter(comets, entropy, color="blue", label="Entropy")
            plt.ylabel("softmax entropy")
            plt.xlabel("reference score")
            plt.savefig(base + model + "entropy.png")
            plt.clf()

            stddiv = [i[2] for i in inps]
            plt.scatter(comets, stddiv, color="blue", label="Standard Deviation")
            plt.ylabel("softmax standard deviation")
            plt.xlabel("reference score")
            plt.savefig(base + model + "stddiv.png")
            plt.clf()

            dropprop = [i[3] for i in inps]
            plt.scatter(comets, dropprop, color="blue", label="Dropout Probability")
            plt.ylabel("dropout translation Probability")
            plt.xlabel("reference score")
            plt.savefig(base + model + "dropprob.png")
            plt.clf()

            dropvar = [i[4] for i in inps]
            plt.scatter(comets, dropvar, color="blue", label="Dropout Variance")
            plt.ylabel("dropout variance")
            plt.xlabel("reference score")
            plt.savefig(base + model + "dropoutvariance.png")
            plt.clf()

            dropoutcombo = [i[5] for i in inps]
            plt.scatter(comets, dropoutcombo, color="blue", label="Dropout Combo")
            plt.ylabel("dropout combo")
            plt.xlabel("reference score")
            plt.savefig(base + model + "dropoutcombo.png")
            plt.clf()
            uniscor = [i[6] for i in inps]
            plt.scatter(uniref, uniscor, color="blue", label="Uniscore")
            plt.ylabel("uniscore")
            plt.xlabel("reference score")
            plt.savefig(base + model + "uniscore.png")
            plt.clf()

            uniscore2 = [i[7] for i in inps]
            plt.scatter(uniref, uniscore2, color="blue", label="Uniscore2")
            plt.ylabel("uniscore with mean")
            plt.xlabel("reference score")
            plt.savefig(base + model + "uniscore2.png")

            plt.clf()
            plt.scatter(x, wers, color="blue", label="Reference")
            plt.xlabel("datasetrows")
            plt.ylabel("wer")
            plt.savefig(base + "werref.png")
        else:
            inps = list(map(float, inpfile.readlines()))
            plt.scatter(x, inps, color="blue", label="Reference")


with open("results/seamlessuniscores.txt", "r") as uniscore:
    lines = uniscore.readlines()
    refs = list(
        map(
            lambda x: (
                float(x.split("\t")[0]),
                float(x.split("\t")[1]),
                float(x.split("\t")[2]),
            ),
            lines,
        )
    )
    alpha = [i[0] / 100 for i in refs]
    pearsoncorr = [i[1] for i in refs]
    pearsoncorr2 = [i[2] for i in refs]
    plt.plot(alpha, pearsoncorr, color="blue", label="transcription probability")
    plt.plot(alpha, pearsoncorr2, color="red", label="transcript probability mean")

    plt.xlabel("alpha")
    plt.ylabel("pearson correlation")
    plt.savefig(base + "seamlessuniscoredistribution.png")
    plt.clf()

with open("results/dlmuniscores.txt", "r") as dlmuniscore:
    lines = dlmuniscore.readlines()
    refs = list(
        map(
            lambda x: (
                float(x.split("\t")[0]),
                float(x.split("\t")[1]),
                float(x.split("\t")[2]),
            ),
            lines,
        )
    )
    alpha = [i[0] / 100 for i in refs]
    pearsoncorr = [i[1] for i in refs]
    pearsoncorr2 = [i[2] for i in refs]
    plt.plot(alpha, pearsoncorr, color="blue", label="transcription probability")
    plt.plot(alpha, pearsoncorr2, color="red", label="transcript probability mean")
    plt.xlabel("alpha")
    plt.ylabel("pearson correlation")
    plt.savefig(base + "dlmuniscoredistribution.png")
    plt.clf()
