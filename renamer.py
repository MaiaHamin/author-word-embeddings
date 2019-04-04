import glob
import os

texts = glob.glob("./personae/data/*.txt")
for text in texts:
    num = os.path.basename(text).split(".")[0]
    newf = "./personae/texts/" + str(num) + ".txt"
    with open(text, "r") as fin:
        with open(newf, "w+") as fout:
            fout.write(fin.read())
