import re

with open("generate-test.txt", "r") as f:
    lines = f.readlines()

    output = [i for i in lines if re.search(r"[TPHSD]-\d.*", i)]

# regex: [TPHSD]-\d.*
