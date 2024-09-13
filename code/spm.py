import sentencepiece as spm
import sys
import os


inputpath = sys.argv[1]
outputpath = sys.argv[2]
spmmodel = sys.argv[3]
print(inputpath, outputpath)
# workspace = os.environ["ws"]
# TEMPDIR = os.environ["TMPDIR"]
sp = spm.SentencePieceProcessor(model_file=spmmodel)
with open(inputpath) as input:
    lines = input.read().split("\n")
    tokens = sp.encode(lines, out_type=str)
    with open(outputpath, "a") as tokenized:
        tokenized.write(tokens)
        tokenized.write("\n")
