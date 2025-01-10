from requests import get
import torch.distributed
from torch.nn.functional import dropout
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_from_disk
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor as SeamlessProcessor
import os
import torch
from tqdm import tqdm
from transformers.models.deprecated.trajectory_transformer import (
    configuration_trajectory_transformer,
)
from qeLogic import getQE, readCSV, writeCSV, TranslationProbability

# dropout would be 0.1 as done in the paper in the experiment for evaluating the translation
model = SeamlessM4Tv2ForTextToText.from_pretrained(
        "facebook/seamless-m4t-v2-large", dropout=0.1, use_cache=False
)
processor = SeamlessProcessor.from_pretrained(
    "facebook/seamless-m4t-v2-large", use_cache=False
)


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=world_size)
    
    elemdp = dataset.num_rows // world_size
    model.to(rank)
    #model = DDP(model)
    model.generation_config.forced_decoder_ids = None
    offset = 0 + rank * elemdp
    csv = []
    with torch.no_grad():
        for i in tqdm(range(offset, offset + elemdp, 1)):
            sample = dataset[i]
            model.eval()
            transcript = sample["reftranscript"]
            
            ####################
            # dropout based shit#
            ####################
            translation = []
            text = processor(
                text=transcript, src_lang="eng", return_tensors="pt").to(rank)
            
            #for j in range(num_samples):
            #############################################
            # Huggingface whisper implementation things #
            #############################################
        
            res = model.generate(
                        **text,
                        tgt_lang="deu",
                        output_scores=True,
                        return_dict_in_generate=True,
            )
            currtranslation = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
            qelist =TranslationProbability(res)
            #print(res, type(res)) 
            #qelist =sum([torch.sum(res.scores[i][torch.nn.functional.argmax(res.scores[i], dim=-1)]) for i in res.scores])/len(res.scores)

            del res
            del text
            torch.cuda.empty_cache()
        
            currrow = [
                i,
                sample["reference"][0],
                sample["reference"][1],
                currtranslation,
                qelist,
            ]
            #currrow.extend(sample["qe"])
            #currrow.extend(sample["transcript"])
            #currrow.extend(qelist)
            #currrow.extend(translation)
            # currrow.extend(dropoutdata)
            #print(currrow)
            writeCSV(currrow, TEMPDIR + "/results/translation" + str(rank)+ ".csv", dropout=False, appen=True)
            del currrow, transcript, qelist
            del sample
            torch.cuda.empty_cache()
    dist.destroy_process_group()

BASE = ""
TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    #dataset = Dataset.from_dict(readCSV(BASE + "results/dropoutfulltranscriptions.csv"))
    #dataset.save_to_disk("seamlesstranslationset")
    dataset = load_from_disk("seamlesstranslationset")
    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)

    run_inference(torchrunrank, world_size, dataset)


if __name__ == "__main__":
    main()


