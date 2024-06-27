
import yaml

#get the data ready for the Models 

def getsegments(): 
    with open("/home/plantpalfynn/uni/BA/BA-Thesis/code/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.yaml", 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        for l in data:

            print(l, "\n", l.get("wav"))


getsegments()
#tar_path = download_asset("dataset/IWSLT23.tst2023.en-de.tar.gz")
#tar_item = "IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/wav/ted_13587.wav"
#with tarfile.open(tar_path, mode="r") as tarfile_:
#    fileobj = tarfile_.extractfile(tar_item)
#    waveform, sample_rate = torchaudio.load(fileobj)