import torch.distributed
from torch.nn.functional import dropout
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, Audio
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor as SeamlessProcessor,
)
import os
import torch
from tqdm import tqdm
from transformers.models.sam.modeling_sam import SamPositionalEmbedding
from qeLogic import getAudios, getQE, writeCSV

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    "facebook/seamless-m4t-v2-large", dropout=0.1, use_cache=False
)
processor = SeamlessProcessor.from_pretrained(
    "facebook/seamless-m4t-v2-large", use_cache=False
)


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    #num = 3
    num = (len(dataset) // (world_size*2))
    #print(num)
    num = 3
    offset = 0 + rank * (num)
    # num = (len(dataset)) // (world_size)
    # print(len(dataset), world_size)
    csv = []
    print("starting seamless regular on", num)

    with torch.no_grad():
        for i in range(offset, offset + num, 1):
            refscore = 0
            model.eval()
            sample = dataset[i]
            model_transctiption = sample["audiofile"]["array"]
            # alternatively set to 16000
            text_input = processor(
                audios=model_transctiption,
                src_lang="eng",
                tgt_lang="deu",
                return_tensors="pt",
                sampling_rate=16000,
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
            #print(model_translation)
            # print(sample["transcript prob"], sample["transcript mean"])
            # torch.cuda.empty_cache()
            # result = (res, model_transctiption, reference_transctipt)
            ## result = Result(sample["audiofile"],sample["timestamp"],sample["transcript"],trans,res,qe)
            # torch.save(result, TEMPDIR + "/results/seamless_result" + str(i) + ".pt")
            torch.cuda.empty_cache()
            dpresults = []
            dropoutres = []
            for i in range(30):
                model.train()
                dropout_input = processor(
                    audios=model_transctiption,
                    src_lang="eng",
                    tgt_lang="deu",
                    return_tensors="pt",
                    sampling_rate=16000,
                )

                dropout_input = dropout_input.to(rank)
                res = model.generate(
                    **dropout_input,
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
                dropout_translation = processor.batch_decode(
                    res["sequences"], skip_special_tokens=True
                )[0]
                # refscore = cometscore([text], [trans], [sample["translation"]])
                dqe = getQE(res, dropout=False, translation=True)
                dpresults.append(res)
                dropoutres.append((dropout_translation, dqe))
            # csv overview: row, model transcript transcripttion reference , modeltranslation, translation reference, qe
            dropoutqe = getQE(dpresults, dropout=True, translation=True)
            print(dropoutqe)
            row = [
                i,
                sample["transcript"],
                sample["translation"], 
                model_translation,
                qe,
                dropoutqe,
            ]
            row.extend(dropoutres)
            csv.append(row)
    #print("done", str(rank))
    #output = [None for _ in range(world_size)]
    #dist.gather_object(
    #    obj=csv, object_gather_list=output if dist.get_rank() == 0 else None, dst=0
    #)
    #if rank == 0:
    #    for i in range(len(output)):
    #        if i == 0:
    #            continue
    #        csv.extend(output[i])
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
    writeCSV(csv, TEMPDIR + "/results/seamlesse2efulltranscriptions"+str(rank)+".csv", dropout=False)
    print("writing done", str(rank))

BASE = ""

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
    # smp = mp.get_context("spawn")
    # q = smp.SimpleQueue()
    # q.put([["sample", "reference", "reference"]])
    mp.spawn(run_inference, args=(world_size, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
