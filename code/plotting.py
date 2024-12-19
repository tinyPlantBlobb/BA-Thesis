import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument("--inp", type=str, default="inp.txt")
args.add_argument("--ref", type=str, default="ref.txt")
ref = args.parse_args().ref
input = args.parse_args().inp
with open(ref, "r") as reffile:
    refs = list(map(float, reffile.readlines()))
    x = range(len(refs))
    plt.scatter(x, refs, color="yellow", label="Reference")

with open(input, "r") as inpfile:
    inps = list(map(float, inpfile.readlines()))
    plt.scatter(x, inps, color="blue", label="Reference")

plt.xlabel("dataset lines")
plt.ylabel("probability")
plt.show()
