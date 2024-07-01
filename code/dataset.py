
import yaml
import xml.etree.ElementTree as ET


#get the data ready for the Models 
BASE = "/home/plantpalfynn/uni/BA/BA-Thesis/code/"
# BASE = "/"

with open(BASE +"dataset/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023/IWSLT.TED.tst2023.en-de.yaml", 'r') as file:
    timestamps = yaml.load(file, Loader=yaml.FullLoader)

allwavs = list(set([l.get("wav") for l in timestamps]))
timestamps = {wav: [elem for elem in timestamps if elem.get("wav") == wav] for wav in allwavs}

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

            print(doc.attrib, i)
            if doc.attrib != {}:
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