import sentencepiece as spm
import sys
import os


inputpath = sys.argv[1]
outputpath = sys.argv[2]
workspace = os.environ["ws"]
TEMPDIR = os.environ["TMPDIR"]
sp = spm.SentencePieceProcessor(model_file=os.environ["SPMMODEL"])
with open(TEMPDIR + inputpath) as input:
    lines = input.read().split("\n")
    tokens = sp.encode(lines, out_type=str)
    with open(TEMPDIR + outputpath, "a") as tokenized:
        tokenized.write(tokens + "\n")
