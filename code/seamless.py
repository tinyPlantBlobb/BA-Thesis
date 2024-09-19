from datasets.features import translation
from requests import get
import torch.distributed
from torch.nn.functional import dropout
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor as SeamlessProcessor
import os
import torch
from tqdm import tqdm
from transformers.models.deprecated.trajectory_transformer import (
    configuration_trajectory_transformer,
)
from qeLogic import getQE, readCSV, writeCSV

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForTextToText.from_pretrained(
    "facebook/seamless-m4t-v2-large", dropout=0.1, use_cache=False, num_beams=5
)
processor = SeamlessProcessor.from_pretrained(
    "facebook/seamless-m4t-v2-large", use_cache=False
)


def run_inference(rank, world_size, dataset):
    # TODO make it work with the distributed data on the different gpus, aka figure out which rank to use and make it work
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    num_samples = 30
    elemdp = dataset.num_rows // world_size
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank * elemdp
    csv = []
    with torch.no_grad():
        for i in range(offset, offset + elemdp, 1):
            sample = dataset[i]
            qetranscript = [sample["qe"][j] for j in range(30)]
            maxqe = qetranscript.index(max(qetranscript, key=(lambda x: x[1])))

            transcript = sample["transcript"][maxqe]
            # print(transcript, sample["reference"])
            # text = sample["reference"]
            ####################
            # dropout based shit#
            #####################
            dropoutdata = []
            translation = []
            qelist = []
            for j in tqdm(range(num_samples)):
                #############################################
                # Huggingface whisper implementation things #
                #############################################
                model.train()
                with torch.no_grad():
                    # this will return the last layer probabilities of the model
                    input = processor(
                        text=transcript, src_lang="eng", return_tensors="pt"
                    )
                    # print(input)
                    input_features = input.to(rank)
                    res = model.generate(
                        **input_features,
                        tgt_lang="deu",
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                    )
                    dropoutdata.append(res)
                    currtranslation = processor.batch_decode(
                        res["sequences"], skip_special_tokens=True
                    )[0]
                    qelist.append(getQE(res, dropout=False, translation=True))
                    translation.append(currtranslation)
                torch.cuda.empty_cache()
            qe = getQE(dropoutdata, dropout=True)
            # with open(TEMPDIR + "/results/dropresult"+str(i)+".txt", "w") as file:
            #     file.write(str(dropoutresult["all"]["tranlateion"]))
            #     file.close()
            # torch.save(dropoutdata, TEMPDIR + "/results/dropoutresult" + str(i) + ".pt")
            currrow = [
                i,
                sample["reference"][0],
                sample["reference"][1],
                qe,
            ]
            currrow.extend(sample["qe"])
            currrow.extend(sample["transcript"])
            currrow.extend(qelist)
            currrow.extend(translation)
            # currrow.extend(dropoutdata)
            csv.append(currrow)
            torch.cuda.empty_cache()
            output = [None for _ in range(world_size)]
            dist.gather_object(
                obj=csv,
                object_gather_list=output if dist.get_rank() == 0 else None,
                dst=0,
            )
            if rank == 0:
                for i in range(len(output)):
                    if i == 0:
                        continue
                    csv.extend(output[i])

                writeCSV(
                    csv,
                    TEMPDIR
                    + "/results/dropouttranslationfulldropped"
                    + str(rank)
                    + ".csv",
                    dropout=True,
                )


BASE = ""

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    dataset = Dataset.from_dict(readCSV(BASE + "results/dropoutfulltranscriptions.csv"))
    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
