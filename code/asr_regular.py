import csv
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import os
import torch
from tqdm import tqdm
from qeLogic import getAudios, getQE, writeCSV


class Result:
    audiofile = None
    timestamp = None
    runs = None
    ref = None
    trans = None
    data = None  # result of the model
    results = None  # Tuple of (qe, qeent, qestd)
    dropoutdata = None  # result of the model for all droutout runs list of tuples
    dropoutresults = None  # list of Tuple of (qe, var, lex-simm)

    def __init__(
        self,
        audiofile,
        timestamp,
        reference,
        transcription,
        modeldata,
        qualityestimate,
        dropoutdata=None,
        dropoutresults=None,
    ):
        self.audiofile = audiofile
        self.timestamp = timestamp
        self.ref = reference
        self.trans = transcription
        self.data = modeldata  # result data of the model
        self.results = qualityestimate  # Tuple of (qe, qeent, qestd)
        self.dropoutresults = dropoutresults
        self.dropoutdata = dropoutdata

    def __str__(self):
        return str(self.trans)

    def __repr__(self):
        return  # "audiofile: "+str(self.audiofile)+ "\n" + "timestamp: "+str(self.timestamp)+ "\n" + "ref: "+str(self.ref)+ "\n"


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank * ((len(dataset)) // world_size)
    num = 3#(len(dataset)) // (world_size * 2)
    csv = []
    with torch.no_grad():
        for i in tqdm(range(offset, offset + num, 1)):
            model.eval()
            sample = dataset[i]
            audio = sample["audiofile"]["array"]
            sample_rate = sample["audiofile"][
                "sampling_rate"
            ]  # alternatively set to 16000
            text = sample["transcript"]
            input = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = input.input_features.to(rank)
            res = model.generate(
                input_features=input_features,
                return_dict_in_generate=True,
                output_scores=True,
            )
            #############################################
            # Huggingface whisper implementation things #
            #############################################

            # this will return the last layer probabilities of the model

            logits = model(
                input_features, decoder_input_ids=res["sequences"]
            ).logits  # gets the last layer probabilities of the model
            trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[
                0
            ]
            qe = getQE(res, dropout=False)
            torch.cuda.empty_cache()
            result = Result(
                sample["audiofile"],
                sample["timestamp"],
                sample["transcript"],
                trans,
                res,
                qe,
            )
            torch.save(result, TEMPDIR + "/results/result" + str(i) + ".pt")
            torch.cuda.empty_cache()

            print(trans, text)
            csv.append([i, text, trans, qe])
    output = [None for _ in range(world_size)]
    dist.gather_object(
        obj=csv, object_gather_list=output if dist.get_rank() == 0 else None, dst=0
    )
    if rank == 0:
        for i in range(len(output)):
            if i == 0:
                continue
            csv.extend(output[i])
        writeCSV(csv, TEMPDIR + "/results/fulltranscriptions.csv", dropout=False)


BASE = ""

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    dataset = Dataset.from_dict(getAudios(BASE)).cast_column("audiofile", Audio())
    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    smp = mp.get_context("spawn")
    q = smp.SimpleQueue()
    q.put([["sample", "reference", "reference"]])
    mp.spawn(run_inference, args=(world_size, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
