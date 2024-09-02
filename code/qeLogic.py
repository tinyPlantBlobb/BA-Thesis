import os
import csv
from re import I
import torch.distributed
import torch.utils
import torch.utils.data
import numpy as np
import torch

import evaluate
import yaml
import torchaudio


def getAudios(TEMPDIR):
    print("starting reading from tar")
    with open(TEMPDIR + "/data/IWSLT.TED.tst2023.en-de.matched.yaml") as matched:
        data = yaml.load(matched, Loader=yaml.FullLoader)
        matched.close()
    print("closed tar")

    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    print("starting iterating over tar elements")
    for t in os.scandir(TEMPDIR + "/data"):
        if t.name == "IWSLT.TED.tst2023.en-de.matched.yaml":
            continue
        tedwav = t.name.split(".")[0]
        segment = int(t.name.split(".")[1][3:])
        seg = data[tedwav + ".wav"][segment]
        torchaudio.load(t.path)
        resdataset["audiofile"].append(t.path)
        resdataset["transcript"].append(seg.get("transcript"))
        resdataset["translation"].append(seg.get("translation"))
        resdataset["timestamp"].append(
            (seg.get("offset"), seg.get("offset") + seg.get("duration"))
        )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    print("finished iterating over elements")
    return resdataset


def TranslationProbability(data):
    # toptoken= data[i].scores
    prop = 0
    for j in range(len(data.scores)):
        toptoken = torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        toptokenprob = torch.log_softmax(data.scores[j][0], dim=-1)[toptoken]
        prop = toptokenprob + prop
    return np.divide(prop.cpu().numpy(), len(data.scores))


def TranscriptionProbability(data):
    prop = 0
    for j in range(len(data.scores)):
        toptoken = torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        toptokenprob = torch.log_softmax(data.scores[j][0], dim=-1)[toptoken]
        prop = toptokenprob + prop
    return prop.cpu().numpy()


def TranscriptionMean(data):
    prop = 0
    for j in range(len(data.scores)):
        toptoken = torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        toptokenprob = torch.log_softmax(data.scores[j][0], dim=-1)[toptoken]
        prop = toptokenprob + prop
    return np.divide(prop.cpu().numpy(), len(data.scores))


def softmaxEntropy(data):
    # Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    prop = 0
    resprop = 0
    for j in range(len(data.scores)):
        softmaxed = torch.softmax(data.scores[j], dim=-1)

        mask = softmaxed != 0
        # logged[mask] = torch.log(softmaxed[mask])
        prop = torch.sum(torch.mul(softmaxed[mask], torch.log(softmaxed[mask])), dim=-1)
        # print("prob", prop, "masked", softmaxed[mask])
        # print("softmax", softmaxed[0], type(softmaxed[0]))
        resprop += prop
        # print(prop)
    qeent = -torch.div(
        resprop, torch.tensor(len(data.scores)).to(torch.distributed.get_rank())
    )

    # qeent = -np.divide(resprop.cpu().numpy(), (len(data.scores)))
    # print("torch", qeee, "np", qeent)
    return qeent.cpu().numpy().item()


def sentStd(data):
    # TODO fix
    # Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    sequence = []
    prop = 0
    for j in range(len(data.scores)):
        toptoken = torch.argmax(torch.nn.functional.softmax(data.scores[j], dim=-1))
        proba = torch.sum(
            torch.max(torch.log_softmax(data.scores[j], dim=-1), dim=1).values
        )
        prop = torch.log_softmax(data.scores[j][0][toptoken], dim=-1) + prop
        sequence.append((prop.cpu(), proba.cpu()))
    # print(sequence)
    qestd = np.std(np.array(sequence))

    return qestd


def writeCSV(results, path, dropout=False):
    if dropout:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, dialect="excel")
            #writer.writerow(["row", "reference", "transcription", "translation", "qe"])
            # writer.writerow(["reference", "transcriptions"])
            writer.writerows(results)
    else:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, dialect="excel")
            writer.writerow(["row", "reference transcript", "reference translation", "transcription", "translation", "transcript prob", "transcript mean","qe"])
            # writer.writerow(["reference", "transcription"])
            writer.writerows(results)


def readCSV(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(
            f,
            dialect="excel",
            fieldnames=["row", "reference transcript", "reference translation", "transcription", "translation", "transcript prob", "transcript mean","qe"],
        )
        data = {"transcript": [], "reference": [], "qe": []}
        for row in reader:
            data["transcript"].append(row["transcript"])
            data["reference"].append(row["reference"])
            data["qe"].append(row["qe"])
    return data


def variance(data):
    print(type(data), type(data[0]))
    data = data[0]
    return torch.var(data, dim=1)


def combo(tp, var):
    return 1 - np.divide(tp, var)


def lexsim(transhypo):
    meteor = evaluate.load("meteor")
    for i in range(len(transhypo) - 1):
        meteor.compute(predictions=transhypo[i], reference=transhypo[i + 1])
    # TODO write code for the simmilarity with the help of meteor

    return 0


def getQE(data, dropout=False, dropouttrans=None, translation=True, ref=0):
    if dropout:
        qe = qevar = lex = []
        com = lex = 0
        for i in range(len(data)):
            if translation:
                qe.append(TranslationProbability(data))
                qevar.append(variance(data))
                com = combo(qe, qevar)
                lex = lexsim(dropouttrans)

            else:
                qe.append(TranscriptionProbability)
        res = (qe, qevar, com, lex)
    else:
        if translation:
            qe = TranslationProbability(data)
            qeent = softmaxEntropy(data)
            qestd = sentStd(data)

            res = (qe, qeent, qestd)
            print(
                "TP",
                res[0],
                "softmaxEntropy",
                res[1],
                "standard deviation",
                res[2],
            )

        else:
            qe = TranscriptionProbability(data)
            qemean = TranscriptionMean(data)
            res = (qe, qemean)
        print(res)
        return res


def cometscore(source, prediction, reference):
    comet_metric = evaluate.load("comet")
    comet_score = comet_metric.compute(
        predictions=prediction, references=reference, sources=source
    )
    print(comet_score)
    return comet_score


def pearsoncorr(prediction, reference):
    pearson_met = evaluate.load("pearsonr")
    result = pearson_met.compute(predictions=prediction, references=reference)
    return result


# TranslationProbability(t)
# softmaxEntropy(t)
# print(t.transcription)
