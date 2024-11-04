import re
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr, worderror
import os
import csv
    
    with open(TMPDIR + "/results/dropoutfulltranslationfulldropped", "r") as resfile:
        reader = csv.DictReader(
            resfile,
            dialect="excel",
            fieldnames=[
            "row",
            "reference transcript",
            "reference translation",
            "transcription",
            "translation",
            "transcript prob",
            "transcript mean",
            "qe",
            ],
        )
        transcripts = []

        translation = []
        reference_translation = []
        reference_trancsript = []
        transcriptionprob = []
        transcriptmean = []
        tpscore = []
        softmaxent = []
        stddiv = []
        for r in reader:
            if r["qe"] != "qe":
            transcripts.append(r["transcription"])
            translation.append(r["translation"])
            reference_translation.append(r["reference translation"])
            reference_trancsript.append(r["reference transcript"])
            # print(r["qe"][1:-1].split(", "))
            qe = r["qe"][1:-1].split(", ")
            transcriptionprob.append(r["transcript prob"])
            print(r["transcript mean"], " probability  ", r["transcript prob"])
            transcriptmean.append(r["transcript mean"])
