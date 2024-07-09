
from os import close
import yaml
import xml.etree.ElementTree as ET


#get the data ready for the Models 
#BASE = "$HOME"
#BASE = "/home/plantpalfynn/uni/BA-Thesis/code"
BASE = "/home/plantpalfynn/uni/BA/BA-Thesis/code"

with open(BASE +"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.yaml", 'r') as file:
    timestamps = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
with open(BASE +"/dataset/IWSLT23.tst2023.en-de/FILE_ORDER", 'r') as wavlist:
    wavlist = wavlist.readlines()
    allwavs = [l.strip() for l in wavlist]
    wavlist.close()

timestamps = {wav+".wav": [elem for elem in timestamps if elem.get("wav") == wav+".wav"] for wav in allwavs}
#XML PART 
detree = ET.parse(BASE +'/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.de.xml')
entree = ET.parse(BASE +'/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.en.xml')
deroot = detree.getroot()
enroot = entree.getroot()

for child in enroot:
    for refset in child:
        
        id = refset.attrib['docid']
        print(refset.attrib, id)
        for i,doc in enumerate(refset):

            print(doc.attrib, i, "\n next thing  \n")
            if doc.attrib != {}:
               print(id)
               print(timestamps["ted_"+id+".wav"][i-1] , doc.attrib, doc.text)

               timestamps["ted_"+id+".wav"][i-1]['transcript']= doc.text
for child in deroot:
    for refset in child:
        id = refset.attrib['docid']
        print(refset.attrib, id)
        for i,doc in enumerate(refset):
            print(doc.attrib, i)
            if doc.attrib != {}:
               print(timestamps["ted_"+id+".wav"][i] , doc.attrib, doc.text)

               timestamps["ted_"+id+".wav"][i]['translation']= doc.text

# DUmping everything into the yaml file

with open(BASE +"/dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.matched.yaml", 'w') as file:
    yaml.dump(timestamps, file)
    file.close()