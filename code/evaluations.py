import re
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr, worderror
import os
import csv

TMPDIR = os.environ["TMPDIR"]

with open(TMPDIR + "/results/seamlessfulltranscriptions.csv", "r", newline="") as file:
    reader = csv.DictReader(
        file,
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
            print(r["qe"][1:-1].split(", "))
            qe = r["qe"][1:-1].split(", ")
            transcriptionprob.append(r["transcript prob"])
            transcriptmean.append(r["transcript mean"])
            tpscore.append(qe[0])
            softmaxent.append(qe[1])
            stddiv.append(qe[2])
    refscores = cometscore(transcripts, translation, reference_translation)

    refscore = refscores["scores"]
    # with open(TMPDIR + "/resultscore.csv", "w") as resscorefile:
    #    reswriter = csv.writer(resscorefile, dialect="excel")
    #    reswriter.writerow(
    #        [
    #            "row",
    #            "transcript",
    #            "reference",
    #            "translation",
    #            "qe tp",
    #            "qe softent",
    #            "qe stddiv",
    #            "refscore",
    #        ]
    #    )
    #    for i in range(len(translation) - 1):
    #        print(i)
    #        reswriter.writerow(
    #            [
    #                i,
    #                trans[i],
    #                reference[i],
    #                translation[i],
    #                tpscore[i],
    #                softmaxent[i],
    #                stddiv[i],
    #                refscore[i],
    #            ]
    #        )
    #    resscorefile.close()
    # print(tpscore)
    # print(refscore)
    wer = worderror(transcripts, reference_trancsript)

    transcriptresult = pearsoncorr(transcriptionprob, wer)
    meanresult = pearsoncorr(transcriptmean, wer)
    tpresult = pearsoncorr(tpscore, refscore)
    softres = pearsoncorr(softmaxent, refscore)
    stdres = pearsoncorr(stddiv, refscore)
    with open(TMPDIR + "/results/scores.txt", "w") as resfile:
        # resfile.write("reference scores\n")
        # resfile.write(str(refscore))
        resfile.write("\n\n\n pearsoncorr of the translation probability")
        resfile.write(str(tpresult))
        resfile.write("\nsoftmax correlation\n")
        resfile.write(str(softres))
        resfile.write("\n standart div\n")
        resfile.write(str(stdres))
        resfile.close()
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
