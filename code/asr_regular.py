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
from qeLogic import getAudios, getQE, writeCSV, writedict


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank * ((len(dataset)) // world_size)
    # num = 3
    num = (len(dataset)) // (world_size)
    csv = []
    with torch.no_grad():
        for i in tqdm(range(offset, offset + num, 1)):
            model.eval()
            sample = dataset[i]
            audio = sample["audiofile"]["array"]
            sample_rate = sample["audiofile"][
                "sampling_rate"
            ]  # alternatively set to 16000
            transcript_reference = sample["transcript"]
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
            # gets the last layer probabilities of the model
            generated_transcript = processor.batch_decode(
                res["sequences"], skip_special_tokens=True
            )[0]
            qe = getQE(res, dropout=False, translation=False)
            torch.cuda.empty_cache()
            # print(qe, generated_transcript, transcript_reference)
            csv.append(
                [
                    i,
                    transcript_reference,
                    sample["translation"],
                    generated_transcript,
                    "",
                    qe[0],
                    qe[1],
                    "",
                ]
            )
    output = [None for _ in range(world_size)]
    dist.gather_object(
        obj=csv, object_gather_list=output if dist.get_rank() == 0 else None, dst=0
    )
    if rank == 0:
        # csv.insert(
        #    0,
        #    [
        #        "row",
        #        "reference transcript",
        #        "reference translation",
        #        "transcription",
        #        "transcript prob",
        #        "transcript mean",
        #    ],
        # )
        for i in range(len(output)):
            if i == 0:
                continue
            csv.extend(output[i])

        writeCSV(csv, TEMPDIR + "/results/fulltranscriptions.csv", dropout=False)
        writedict(
            TEMPDIR,
            generated_transcript=[csv[i][3] for i in range(len(csv))],
            transcription_reference=[csv[i][1] for i in range(len(csv))],
            translation_reference=[csv[i][2] for i in range(len(csv))],
        )
        print(TEMPDIR + "/results/fulltranscriptions.csv")


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
