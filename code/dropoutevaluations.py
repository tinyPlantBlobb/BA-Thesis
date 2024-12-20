import re
from code import qeLogic
from datasets.features import translation
from qeLogic import cometscore, pearsoncorr, worderror
import os
import csv
import math

TMPDIR = "/home/plantpalfynn/uni/BA/BA-Thesis/code/"
transcrptqes = ["transcript probability " + str(i) for i in range(0, 30)]
# TMPDIR = os.environ["TMPDIR"]
row = [
    "row",
    "reference transcript",
    "reference translation",
    "regulartranslation",
    "qe",
]
transcriptindex = ["transcript probability " + str(i) for i in range(0, 30)]
row.extend(["transcript probability " + str(i) for i in range(0, 30)])
row.extend(["transcript " + str(i) for i in range(0, 30)])
row.extend(["translation probability" + str(i) for i in range(0, 30)])
translationindex = ["translation probability " + str(i) for i in range(0, 30)]
row.extend(["translation " + str(i) for i in range(0, 30)])
with open(TMPDIR + "/results/translation0.csv", "r", newline="") as file:
    reader = csv.DictReader(
        file,
        dialect="excel",
        fieldnames=row,
    )
    transcripts = []
    translation = []
    reference_translation = []
    reference_trancsript = []
    transcriptionprob = []
    transcriptmean = []
    transcrptprobabiltiy = []
    translationEntropy = []
    translationstddiv = []
    translationProbability = []
    translationqe = []
    translationdpprob = []
    translationdpvarinace = []

    tpscore = []
    softmaxent = []
    stddiv = []

    for r in reader:
        # print(r)
        if r["qe"] != "qe":
            transcripts.append(r["transcription"])
            translation.append(r["regulartranslation"])
            reference_translation.append(r["reference translation"])
            reference_trancsript.append(r["reference transcript"])
            # print(r["qe"][1:-1].split(", "))
            qe = re.findall(r"-?\d+\.\d+", r["qe"])
            # transcriptionprob.append(r["transcript prob"])
            transcriptionprob.append(
                [
                    [float(j.strip()) for j in r[i].strip("()").split(",")]
                    for i in transcriptindex
                ]
            )

            print(r["transcript mean"], " probability  ", r["transcript prob"])
            transcriptmean.append([t[1] for t in transcriptionprob])
            transcrptprobabiltiy.append([i[0] for i in transcriptionprob])
            qes = [
                list(map(float, r[i].strip("()").split(","))) for i in translationindex
            ]

            translationqe.append(qes)
            translationdpprob.append(math.fsum([i[0] for i in qes]))
            translationdpvarinace.append(qeLogic.variance([i[0] for i in qes]))

    refscores = cometscore(transcripts, translation, reference_translation)

    refscore = refscores["scores"]
    with open(TMPDIR + "referencescores.txt", "w") as export:
        for i in refscore:
            export.write(i)
            export.write("\n")
    wer = [
        worderror([transcripts[i].lower()], [reference_trancsript[i].lower()])
        for i in range(len(transcripts))
    ]
    # print(wer, len(wer), len(transcriptmean))
    # print(transcriptmean)
    meanresult = pearsoncorr(transcriptmean, wer)
    tpresult = pearsoncorr(tpscore, refscore)
    softres = pearsoncorr(softmaxent, refscore)
    stdres = pearsoncorr(stddiv, refscore)
    transcriptresult = pearsoncorr(transcriptionprob, wer)
    print(transcriptresult, tpresult, softres, stdres)
    with open(TMPDIR + "/results/scores.txt", "w") as resfile:
        # resfile.write("reference scores\n")
        # resfile.write(str(refscore))
        resfile.write("\n\n\n pearsoncorr of the translation probability")
        resfile.write(str(tpresult["pearsonr"]))
        resfile.write("\nsoftmax entropy correlation\n")
        resfile.write(str(softres["pearsonr"]))
        resfile.write("\n standart div\n")
        resfile.write(str(stdres["pearsonr"]))
        resfile.write("\n transcription result \n")
        resfile.write(str(transcriptresult["pearsonr"]))
        resfile.write("\n trasncript mean\n")
        resfile.write(str(meanresult["pearsonr"]))
        resfile.close()
    # for row in reader:
    #    score = cometscore(row["transcript"], row["translation"], row["reference"])
    #    print(score, row["qe"])
    # from datasets.features import translation
    # from qeLogic import pearsoncorr, worderror, cometscore
    # import os
    # import csv
    #
    # world_size = 4
    ## world_size = torch.cuda.device_count()
    # transcripts = []
    #
    # translation = []
    # reference_translation = []
    # reference_trancsript = []
    # transcriptionprob = []
    # transcriptmean = []
    # tpscore = []
    # softmaxent = []
    # stddiv = []
    # fullthing = []
    # for i in range(world_size):
    #    with open(
    #            TMPDIR + "results/dropouttranslationfulldropped" + str(i) + ".csv",
    #        "r",
    #    ) as resfile:
    #        reader = csv.DictReader(
    #            resfile,
    #            dialect="excel",
    #            fieldnames=row,
    #        )
    #        for r in reader:
    #            if r["row"] != "row":
    #                # print(type(r))
    #                fullthing.append(r)
    #
    #
    # with open("seamlessdropout.csv", "w") as csvfile:
    #    writer = csv.DictWriter(csvfile, fieldnames=row)
    #    writer.writerows(sorted(fullthing, key=lambda x: int(x["row"])))
    # reference_scores =
    #
    ##with open("referencescores.txt", "r") as ref:
    ##    lines = ref.readlines()
    ##    reference_scores = [float(i) for i in lines]
    #
    # with open(("seamlessdropout.csv"), "r") as csvfile:
    #    reader = csv.DictReader(csvfile, fieldnames=row)
    #    for row in reader:
    #        # transcripts.append(r["transcription"])
    #        translation.extend([row[i] for i in transcriptindex])
    #        reference_translation.append(row["reference translation"])
    #        reference_trancsript.append(row["reference transcript"])
    #        # print(r["qe"][1:-1].split(", "))
    #        # qe = r["qe"][1:-1].split(", ")
    #        transcriptionprob.append(
    #            [[float(j) for j in re.findall(r"-?\d+\.\d*", row[i])] for i in qes]
    #        )
    #    print(len(transcriptionprob))
    #    cutoff = len(transcriptionprob)
    #    translationProbability = [sum([j[0] for j in i]) / 30 for i in transcriptionprob]
    #
    #    # softmaxent = [i[1] for i in transcriptionprob]
    #    # stddiv = [i[2] for i in transcriptionprob]
    #    result = pearsoncorr(translationProbability, reference_scores[:cutoff])
    #    # result2 = pearsoncorr(softmaxent, reference_scores[:cutoff])
    #    # result3 = pearsoncorr(stddiv, reference_scores[:cutoff])
    #    print(result)
