import os
import csv
import torch.distributed
import torch.utils
import torch.utils.data
import numpy as np
import torch
from comet import download_model, load_from_checkpoint
import evaluate
import yaml
import torchaudio


def getAudios(TEMPDIR):
    # print("starting reading from tar")
    with open(TEMPDIR + "data/IWSLT.TED.tst2023.en-de.matched.yaml") as matched:
        data = yaml.load(matched, Loader=yaml.FullLoader)
        matched.close()
    

    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    # print("starting iterating over tar elements")
    for t in os.scandir(TEMPDIR + "data"):
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
    # print("finished iterating over elements")
    return resdataset


def worderror(hypothesis, references):
    wer = evaluate.load("wer")
    wer_score = wer.compute(predictions=hypothesis, references=references)
    return wer_score


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
    return prop.cpu().numpy().astype(float).item()


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
        resprop += prop
        # print(prop)
    qeent = -torch.div(
        resprop, torch.tensor(len(data.scores)).to(torch.distributed.get_rank())
    )

    # qeent = -np.divide(resprop.cpu().numpy(), (len(data.scores)))
    # print("torch", qeee, "np", qeent)
    return qeent.cpu().numpy().item()


def stddiv(data):
    return np.std(np.array(data))


def sentStd(data):

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


def writeCSV(results, path, dropout=False, appen=False):
    if dropout:
        if not appen:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f, dialect="excel")
                row = [
                    "row",
                    "reference transcript",
                    "reference translation",
                    "regulartransaltion",
                    "qe",
                ]

                row.extend(["transcript probability " + str(i) for i in range(30)])
                row.extend(["transcript " + str(i) for i in range(30)])
                row.extend(["translation probability"+str(i) for i in range(30)])
                row.extend(["translation" + str(i) for i in range(30)])

                # print(type(results), results)
                writer.writerow(row)
                # writer.writerow(["reference", "transcriptions"])
                #writer.writerows(results)

        else:
            with open(path, "a", newline="") as f:
                writer = csv.writer(f, dialect="excel")
                writer.writerow(results)
    else:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, dialect="excel")
            writer.writerow(
                [
                    "row",
                    "reference transcript",
                    "reference translation",
                    "transcription",
                    "translation",
                    "transcript prob",
                    "transcript mean",
                    "qe",
                ]
            )
            # writer.writerow(["reference", "transcription"])
            writer.writerows(results)


def readCSV(path):
    with open(path, "r", newline="") as f:
        row = [
            "row",
            "reference transcript",
            "reference translation",
            "qe",
        ]
        row.extend(["transcript probability " + str(i) for i in range(30)])
        row.extend(["transcript " + str(i) for i in range(30)])
        probabilities = ["transcript probability " + str(i) for i in range(30)]
        transcpipts = ["transcript " + str(i) for i in range(30)]
        reader = csv.DictReader(
            f,
            dialect="excel",
            fieldnames=row,
        )
        data = {
            "transcript": [],
            "reftranscript": [],
            # "translation reference": [],
            "qe": [],
            "reference": [],
        }
        for row in reader:
            if row["row"] == "row":
                continue
            data["transcript"].append([row[i] for i in transcpipts])

            data["reference"].append(
                (row["reference transcript"], row["reference translation"])
            )
            # data["transcription reference"].append(row["reference transcript"])
            # print("probabilities 0", probabilities[0], row[probabilities[0]])
            #[print(row[i])for i in probabilities]
            qe =[
                    (
                        float(row[i].split(",")[0][1:]),
                        float(row[i].split(",")[1][:-1]),
                    )
                    for i in probabilities
                ]
            data["qe"].append(qe)
            data["reftranscript"].append([row[i] for i in transcpipts][qe.index(max(qe, key=lambda x:x[1]))])
                    
            # print(type(data["qe"][0][0]), "\n")
            # data["qe"].append((row["transcript probability"], row["transcript mean"]))
    return data


def variance(data):
    # print(type(data), type(data[0]))
    return torch.var(torch.as_tensor(data), dim=-1)


def combo(tp, var):
    if var == 0:
        return 0
    return 1 - torch.div(tp, var).cpu().numpy()


def lexsim(transhypo):
    res = []
    meteor = evaluate.load("meteor")
    for j in range(len(transhypo) - 1):
        for i in range(len(transhypo) - 1):
            if i == j:
                continue
            res.append(
                meteor.compute(predictions=transhypo[i], reference=transhypo[i + 1])
            )
    # TODO write code for the simmilarity with the help of meteor
    return torch.mul(
        torch.div(1, torch.div(1, 2) * len(transhypo) * (len(transhypo) - 1)),
        torch.sum(res),
    )


def getQE(data, dropout=False, dropouttrans=None, translation=True):
    if dropout:
        qe = qevar = lex = qemean = []
        com = lex = 0
        if translation:
            for i in range(len(data)):
                # print(data[i])
                qe.append(TranslationProbability(data[i]))
                # lex = lexsim(dropouttrans)
            tpdropout = torch.div(torch.sum(torch.as_tensor(qe)), 30)
            qevar = variance(qe)
            com = combo(tpdropout, qevar)
            res = (qe, qevar, com, lex)
        else:
            for i in range(len(data)):
                # print(type(data[i]), data[i], type(data))
                qe.append(TranscriptionProbability(data[i]))
                qemean.append(TranscriptionMean(data[i]))
            qevar.append(variance(qe))
            res = (qe, qemean, qevar)
    else:
        if translation:
            qe = TranslationProbability(data)
            qeent = softmaxEntropy(data)
            qestd = sentStd(data)

            res = (qe, qeent, qestd)
            # print(
            #    "TP",
            #    res[0],
            #    "softmaxEntropy",
            #    res[1],
            #    "standard deviation",
            #    res[2],
            # )
        else:
            qe = TranscriptionProbability(data)
            qemean = TranscriptionMean(data)
            res = (qe, qemean)

        return res


def cometscore(source, prediction, reference):
    comet_metric = evaluate.load("comet")

    comet_metric.add_batch(predictions=prediction, references=reference, sources=source)
    comet_score = comet_metric.compute()
    
    return comet_score
def comet(source, prediction, reference):
    data=[{"src":source[i], "mt":prediction[i], "ref":reference[i]} for i in range(len(source))]
    model_path=download_model("Unbabel/wmt22-comet-da")
    model =load_from_checkpoint(model_path)
    res = model.predict(data, batch_size=8,gpus=torch.cuda.device_count())
    print(res)
    return res

def pearsoncorr(prediction, reference):
    pearson_met = evaluate.load("pearsonr")
    result = pearson_met.compute(predictions=prediction, references=reference)
    return result


def writedict(
    TEMPDIR, generated_transcript, transcription_reference, translation_reference
):
    with open(
        "/pfs/work7/workspace/scratch/utqma-finals/data-bin/test.en", "w"
    ) as src:
        for i in range(len(generated_transcript)):
            src.write(generated_transcript[i])
            src.write("\n")
        src.close()
    with open(
        "/pfs/work7/workspace/scratch/utqma-finals/data-bin/test.de", "w"
    ) as tgt:
        for i in range(len(translation_reference)):
            tgt.write(translation_reference[i])
            tgt.write("\n")
        src.close()
    with open(
        "/pfs/work7/workspace/scratch/utqma-finals/data-bin/ref.de", "w"
    ) as ref:
        for i in range(len(translation_reference)):
            ref.write(translation_reference[i])
            ref.write("\n")
        ref.close()
