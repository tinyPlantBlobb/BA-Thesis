import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument("--inp", type=str, default="inp.txt")
args.add_argument("--ref", type=str, default="ref.txt")
args.add_argument("--split", type=bool, default=False)
ref = args.parse_args().ref
input = args.parse_args().inp
splitting = args.parse_args().split
with open(ref, "r") as reffile:
    refs = list(map(float, reffile.readlines()))
    x = range(len(refs))
    print(len(x), len(refs))
    # plt.scatter(x, refs, color="yellow", label="Reference")
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
            x = range(len(genprop))
            plt.scatter(x, genprop, color="red", label="Probability")
            plt.scatter(x, entropy, color="green", label="Entropy")
            plt.scatter(x, stddiv, color="purple", label="Standard Deviation")
            plt.scatter(x, dropprop, color="orange", label="Drop Probability")
            plt.scatter(x, dropvar, color="black", label="Drop Variance")
            # plt.scatter(x, inps, color="gold", label="Reference")
        else:
            inps = list(map(float, inpfile.readlines()))
            plt.scatter(x, inps, color="blue", label="Reference")

plt.xlabel("dataset lines")
plt.ylabel("probability")
plt.show()
