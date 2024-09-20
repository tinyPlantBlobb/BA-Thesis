import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor as SeamlessProcessor,
)
import os
import torch
from tqdm import tqdm
from qeLogic import getAudios, getQE, writeCSV

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessProcessor.from_pretrained("facebook/seamless-m4t-v2-large")


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    num = 280
    num = dataset.num_rows["train"] // world_size
    print(num)
    # num = 3
    offset = 0 + rank * (num)
    # num = (len(dataset)) // (world_size)
    # print(len(dataset), world_size)
    csv = []
    print("starting seamless regular on", num)

    with torch.no_grad():
        for i in tqdm(range(offset, offset + num, 1)):
            refscore = 0
            model.eval()
            sample = dataset[i]
            model_transctiption = sample["audiofile"]["array"]
            # alternatively set to 16000
            # reference transctiption
            reference_transctipt = sample
            text_input = processor(
                text=model_transctiption,
                src_lang="eng",
                tgt_lang="deu",
                return_tensors="pt",
            )

            text_input = text_input.to(rank)
            res = model.generate(
                **text_input,
                tgt_lang="deu",
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )

            #############################################
            # Huggingface whisper implementation things #
            #############################################

            # this will return the last layer probabilities of the model

            # model(input_features, decoder_input_ids=res["sequences"]).logits  # gets the last layer probabilities of the model
            model_translation = processor.batch_decode(
                res["sequences"], skip_special_tokens=True
            )[0]
            # refscore = cometscore([text], [trans], [sample["translation"]])
            qe = getQE(res, dropout=False)
            # print(model_translation, reference_transctipt, qe)
            # print(sample["transcript prob"], sample["transcript mean"])
            # torch.cuda.empty_cache()
            # result = (res, model_transctiption, reference_transctipt)
            ## result = Result(sample["audiofile"],sample["timestamp"],sample["transcript"],trans,res,qe)
            # torch.save(result, TEMPDIR + "/results/seamless_result" + str(i) + ".pt")
            torch.cuda.empty_cache()
            # csv overview: row, model transcript transcripttion reference , modeltranslation, translation reference, qe

            csv.append(
                [
                    i,
                    reference_transctipt,
                    sample["reference translation"],
                    model_transctiption,
                    model_translation,
                    sample["transcript prob"],
                    sample["transcript mean"],
                    qe,
                ]
            )
    output = [None for _ in range(world_size)]
    dist.gather_object(
        obj=csv, object_gather_list=output if dist.get_rank() == 0 else None, dst=0
    )
    if rank == 0:
        for i in range(len(output)):
            if i == 0:
                continue
            csv.extend(output[i])
        csv.insert(
            0,
            [
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
        writeCSV(
            csv, TEMPDIR + "/results/seamlessfulltranscriptions.csv", dropout=False
        )


BASE = ""

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    print(TEMPDIR + "/results/fulltranscriptions.csv")
    dataset = Dataset.from_dict(getAudios(BASE)).cast_column("audiofile", Audio())

    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    # smp = mp.get_context("spawn")
    # q = smp.SimpleQueue()
    # q.put([["sample", "reference", "reference"]])
    mp.spawn(run_inference, args=(world_size, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
