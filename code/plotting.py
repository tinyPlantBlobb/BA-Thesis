import matplotlib.pyplot as plt
import argparse

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
with open(ref, "r") as reffile:
    lines = reffile.readlines()
    refs = list(
        map(lambda x: (float(x.split("\t")[0]), float(x.split("\t")[1])), lines)
    )
    wers = [i[1] for i in refs]
    comets = [i[0] for i in refs]

    x = range(len(refs))
    # print(len(x), len(refs))
    plt.scatter(x, wers, color="green", label="Reference")
    plt.savefig(base + model + "werref.png")
    plt.clf()
    plt.scatter(x, comets, color="blue", label="Comet")
    plt.savefig(base + model + "references.png")
    plt.clf()
if input != "ref.txt":
    with open(input, "r") as inpfile:
        if splitting:
            lines = inpfile.readlines()
            cases = [l.split(",") for l in lines]
            inps = [list(map(float, i)) for i in cases]
            genprop = [i[0] for i in inps]
            entropy = [i[1] for i in inps]
            stddiv = [i[2] for i in inps]
            dropprop = [i[3] for i in inps]
            dropvar = [i[4] for i in inps]
            print(
                len(x),
                len(genprop),
                len(entropy),
                len(stddiv),
                len(dropprop),
                len(dropvar),
            )
            # x = range(len(genprop))
            plt.scatter(comets, genprop, color="red", label="Probability")
            plt.savefig(base + model + "genprob.png")
            plt.clf()
            plt.scatter(comets, dropprop, color="orange", label="Drop Probability")
            plt.savefig(base + model + "dropprob.png")
            plt.clf()
            plt.scatter(comets, entropy, color="green", label="Entropy")
            plt.savefig(base + model + "entropy.png")
            plt.clf()
            plt.scatter(comets, stddiv, color="purple", label="Standard Deviation")
            plt.savefig(base + model + "stddiv.png")
            plt.clf()
            plt.scatter(comets, dropvar, color="black", label="Drop Variance")
            # plt.scatter(x, inps, color="gold", label="Reference")
        else:
            inps = list(map(float, inpfile.readlines()))
            plt.scatter(x, inps, color="blue", label="Reference")

plt.xlabel("dataset lines")
plt.ylabel("probability")
plt.savefig(base + model + "dropoutvarinace.png")
