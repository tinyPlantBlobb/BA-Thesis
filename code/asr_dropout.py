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


def run_inference(rank, world_size, dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(rank)
    model.generation_config.forced_decoder_ids = None
    num_runs = 30
    #num_runs= 4
    #num = 3
    num = (len(dataset)) // (world_size)
    offset = 0+ rank*num
    if rank == world_size -1:
        #print(len(dataset)-num*world_size)
        num += len(dataset)%world_size
    csv = []
    #print(world_size, rank, offset, num)
    
    
    writeCSV(csv, respath+"transcript"+str(rank)+".csv",dropout=True, appen=False)

    with torch.no_grad():
        for i in tqdm(range(offset, offset+num, 1)):
            model.eval()
            sample = dataset[i]
            audio = sample["audiofile"]["array"]
            sample_rate = sample["audiofile"][
                "sampling_rate"
            ]  # alternatively set to 16000

            transcript_reference = sample["transcript"]
            input = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = input.input_features.to(rank)
            ref=processor.batch_decode(model.generate(input_features=input_features, return_dict_in_generate=True, output_scores=True)["sequences"], skip_special_tokens=True)[0]
            generated_transcripts = []
            generations = []
            qelist = []
            row = []
            model.train()
            for _ in range(num_runs):
                
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
                generations.append(res)
                qelist.append(qe)
                generated_transcripts.append(generated_transcript)
                del res
                # print(generated_transcript, transcript_reference)
                
            #qe = getQE(generations, dropout=True, translation=False)
            #del generations
            row = [
                i,
                transcript_reference,
                sample["translation"],
                ref,
            ]
            row.extend(qelist)
            row.extend(generated_transcripts)
            csv.append(row)
            del qelist
            del generated_transcripts
            writeCSV(row,respath+"/transcript"+str(rank)+".csv",dropout=True, appen=True)
            torch.cuda.empty_cache()
    output = [None for _ in range(world_size)]
    dist.gather_object(
        obj=csv, object_gather_list=output if dist.get_rank() == 0 else None, dst=0
    )

    if rank == 0:
        #        csv.insert(
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

        writeCSV(csv, TEMPDIR + "/dropoutfulltranscriptions.csv", dropout=True)

        

BASE = ""

# Whisper  from the huggingface whisper implementation
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium.en", dropout=0.1
)

TEMPDIR = os.environ["TMPDIR"]
respath = os.path.join(TEMPDIR, "results")
BASE = TEMPDIR + "/"
if not os.path.exists(respath):
    os.mkdir(respath)


def main():
    dataset = Dataset.from_dict(getAudios(BASE)).cast_column("audiofile", Audio())
    print(len(dataset))
    world_size = torch.cuda.device_count()
    torchrunrank = int(os.environ["LOCAL_RANK"])
    trglrank = int(os.environ["RANK"])
    run_inference(torchrunrank, world_size, dataset)


if __name__ == "__main__":
    main()
