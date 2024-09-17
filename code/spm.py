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
    tokens = sp.EncodeAsPieces(lines, out_type=str)
    with open(outputpath, "a") as tokenized:
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                tokenized.write(tokens[i][j])
                tokenized.write(" ")
            tokenized.write("\n")
