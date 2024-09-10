from requests import get
import torch.distributed
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
from qeLogic import getQE, readCSV, writeCS

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessProcessor.from_pretrained("facebook/seamless-m4t-v2-large")


def run_inference(rank, world_size, dataset):
    # TODO make it work with the distributed data on the different gpus, aka figure out which rank to use and make it work
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    num_samples = 30
    elemdp = dataset.num_rows["train"] // world_size
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank * elemdp
    csv = []
    with torch.no_grad():
        for i in range(offset, offset + elemdp, 1):
            sample = dataset["train"][i]
            transcript = sample["transcript"]
            text = sample["tranlation reference"]
            ####################
            # dropout based shit#
            #####################
            dropoutdata = []
            for j in tqdm(range(num_samples)):
                #############################################
                # Huggingface whisper implementation things #
                #############################################
                model.train()
                with torch.no_grad():
                    # this will return the last layer probabilities of the model
                    input = processor(transcript, return_tensors="pt")
                    input_features = input.input_features.to(rank)
                    res = model.generate(
                        input_features=input_features,
                        tgt_lang="deu",
                        return_dict_in_generate=True,
                        output_scores=True,
                        generate_speech=False,
                    )
                    dropoutdata.append(res)
                torch.cuda.empty_cache()
            qe = getQE(dropoutdata, dropout=True)
            # with open(TEMPDIR + "/results/dropresult"+str(i)+".txt", "w") as file:
            #     file.write(str(dropoutresult["all"]["tranlateion"]))
            #     file.close()
            # torch.save(dropoutdata, TEMPDIR + "/results/dropoutresult" + str(i) + ".pt")
            csv.append(
                [
                    i,
                    sample["transcription reference"],
                    sample["translation reference"],
                    qe,
                ].extend(dropoutdata)
            )
        writeCSV(csv, "results/dropouttranslationfull.csv", dropout=True)
        del sample


BASE = ""


TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    dataset = Dataset.from_dict(readCSV(BASE + "results/fulltranscription.csv"))
    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
