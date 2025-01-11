import yaml
import xml.etree.ElementTree as ET
import os

# get the data ready for the Models
BASE = os.environ["HOME"]
def getyaml(BASE):
    with open(
        BASE
        + "/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.yaml",
        "r",
    ) as file:
        timestamps = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    with open(BASE + "/dataset/IWSLT23.tst2023.en-de/FILE_ORDER", "r") as wavs:
        wavlist = wavs.readlines()
        allwavs = [wav.strip() for wav in wavlist]
        wavs.close()
    
    timestamps = {
        wav + ".wav": [elem for elem in timestamps if elem.get("wav") == wav + ".wav"]
        for wav in allwavs
    }
    # XML PART
    detree = ET.parse(
        BASE
        + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.de.xml"
    )
    entree = ET.parse(
        BASE
        + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.en.xml"
    )
    deroot = detree.getroot()
    enroot = entree.getroot()
    
    for child in enroot:
        for refset in child:
            id = refset.attrib["docid"]
            print(refset.attrib, id)
            for i, doc in enumerate(refset):
                print(doc.attrib, i, "\n next thing  \n")
                if doc.attrib != {}:
                    print(id)
                    print(timestamps["ted_" + id + ".wav"][i - 1], doc.attrib, doc.text)
    
                    timestamps["ted_" + id + ".wav"][i - 1]["transcript"] = doc.text
    for child in deroot:
        for refset in child:
            id = refset.attrib["docid"]
            print(refset.attrib, id)
            for i, doc in enumerate(refset):
                print(doc.attrib, i)
                if doc.attrib != {}:
                    print(timestamps["ted_" + id + ".wav"][i], doc.attrib, doc.text)
    
                    timestamps["ted_" + id + ".wav"][i]["translation"] = doc.text
    
    # DUmping everything into the yaml file
    
    with open(
        BASE
        + "/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "w",
    ) as file:
        yaml.dump(timestamps, file)
        file.close()
    
    
    
import torch.distributed
import torch.utils
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import yaml
import tarfile
import torch
import torchaudio
from tqdm import tqdm
def getsegments(BASE):
    with open(
        BASE
        + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml",
        "r",
    ) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data

def segmentAudio(BASE):
    dataset = getsegments(BASE)
    sample_rate = 16000
    audiopath = BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/"
    path = BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    if not os.path.exists(
        BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
    ):
        os.makedirs(
            BASE + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
        )
    for data in dataset:
        for i, seg in enumerate(dataset[data]):
            frame_offset = int(seg.get("offset") * 16000)
            num_frames = int(seg.get("duration") * 16000)
            waveform, sample_rate = torchaudio.load(
                audiopath + seg.get("wav"),
                frame_offset=frame_offset,
                num_frames=num_frames,
                format="wav",
                )
            path = (
                BASE
                + "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
                + seg.get("wav")
                + str(i)
                + ".wav"
            )
            if not os.path.exists(path):
                torchaudio.save(path, waveform, sample_rate)
            resdataset["audiofile"].append(path)
            # resdataset["waveform"].append(waveform)
            resdataset["transcript"].append(seg.get("transcript"))
            # resdataset["audiofile"].append(seg.get("wav"))
            resdataset["translation"].append(seg.get("translation"))
            resdataset["timestamp"].append(
                (seg.get("offset"), seg.get("offset") + seg.get("duration"))
            )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
        with tarfile.open("segmented_IWSLT-23.en-de.zip", "w:gz") as segmented:
            segmented.add(BASE+ "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/segmented/"
, arcname=".")
    return resdataset

def readfromtar(BASE):
    tar = tarfile.open(BASE + "segments_IWSLT-23.en-de.tar.gz", "r:gz")
    with tar.extractfile("IWSLT.TED.tst2023.en-de.matched.yaml") as matched:
        data = yaml.load(matched, Loader=yaml.FullLoader)
        matched.close()
    
    sample_rate = 16000
    resdataset = {
        "audiofile": [],
        "transcript": [],
        "translation": [],
        "timestamp": [],
    }
    for t in tar.getmembers():
        if t.name == "IWSLT.TED.tst2023.en-de.matched.yaml":
            continue
        tedwav = t.name.split(".")[0]
        segment= int(t.name.split(".")[1][3:]) 
        seg = data[tedwav+".wav"][segment]
        print(t)
        resdataset["audiofile"].append(t)
        resdataset["transcript"].append(seg.get("transcript"))
        resdataset["translation"].append(seg.get("translation"))
        resdataset["timestamp"].append(
            (seg.get("offset"), seg.get("offset") + seg.get("duration"))
        )  # falls man noch die xmls rein matchen will: , "transcript":seg.get("text"), "translation":seg.get("translation")
    tar.close()
    return resdataset

segements = segmentAudio(BASE)

#readfromtar(os.environ["HOME"])

