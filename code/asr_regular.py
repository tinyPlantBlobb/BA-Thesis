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
import yaml
import torch
import torchaudio
from tqdm import tqdm
from qeLogic import getAudios, getQE, Result, writeCSV

def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None

    offset = 0 + rank*((len(dataset))//world_size)
    with torch.no_grad():
        for i in tqdm(range(offset, offset+30,1)):
            # result = {
            #     "audiofile": [],
            #     "timestamp": [],
            #     "ref": [],
            #     "generationoutput": [],
            #     "logits": [],
            #     "softmax": [],
            #     "transcription": [],
            #     "qe": [],
            #     "qeent": [],
            # }
            model.eval()
            sample = dataset[i]
            # for sample in tdqm(dataset):
            #print(sample["transcript"])
            audio =sample["audiofile"]["array"]
            sample_rate = sample["audiofile"]["sampling_rate"]  # alternatively set to 16000
            text = sample["transcript"]
            input = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = input.input_features.to(rank)
            res = model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True, output_logits=True)
            #############################################
            # Huggingface whisper implementation things #
            #############################################

            # this will return the last layer probabilities of the model

            logits = model(
                input_features, decoder_input_ids=res["sequences"]
            ).logits  # gets the last layer probabilities of the model
            trans = processor.batch_decode(res["sequences"], skip_special_tokens=True)[0]
            # qe = TranslationProbability(res)
            # qeent = softmaxEntropy(res)
            # qesent = sentStd(res)


            # # # get the frist average log probability of the model for that aucio
            # result["qe"].append(qe)
            # result["qeent"].append(qeent)
            # result["qesent"].append(qesent)
            # result["audiofile"].append(sample["audiofile"])
            # result["timestamp"].append(sample["timestamp"])
            # result["logits"].append(logits)
            # result["softmax"].append(torch.nn.functional.softmax(logits, dim=-1))
            # result["generationoutput"].append(res)
            # result["ref"].append(text)
            # result["transcription"].append(trans+"\n")
            qe= getQE(res, dropout=False)
            torch.cuda.empty_cache()
            result = Result(sample["audiofile"],sample["timestamp"],sample["transcript"],trans,res,qe)
            torch.save(result, TEMPDIR + "/results/result"+str(i)+".pt")
            torch.cuda.empty_cache()
            writeCSV(trans, TEMPDIR + "/results/fulltranscriptions.csv", refence=text, dropout=False)

BASE = ""

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR +"/"
if not os.path.exists(respath):
        os.mkdir(respath)

def main():
    dataset = Dataset.from_dict(getAudios(BASE)).cast_column("audiofile", Audio())
    world_size= torch.cuda.device_count()
    torchrunrank= int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    print("start rank", torchrunrank, trglrank)
    mp.spawn(run_inference, args=(world_size,dataset),nprocs=world_size, join=True)
if __name__ == "__main__":
    main()

