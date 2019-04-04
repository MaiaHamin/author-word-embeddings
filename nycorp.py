import xml.etree.ElementTree as ET
from nltk.corpus.reader import XMLCorpusReader
import os
import glob
from collections import defaultdict

root1 = './nyt_corpus/data/2005/**/**/'
root2 = './nyt_corpus/data/2004/**/**/'
text1 = glob.glob(root1 + '*.xml')
text2 = glob.glob(root2 + '*.xml')
texts = text1

authlist = ['bob herbert',
'david brooks',
'nicholas d. kristof',
'thomas l. friedman',
'paul krugman',
'maureen dowd',
'frank rich',
'verlyn klinkenborg',
'adam cohen',
'lawrence downes']

nottestmode = False

authord = defaultdict(list)

icount = 0
ncount = 0
acount = 0
for filename in texts:
    reader = XMLCorpusReader(os.path.dirname(filename), os.path.basename(filename))
    xml = reader.xml()
    ptext = ""
    desk = ""
    body = xml.find('body')
    head = xml.find('head')
    auth = body.find('body.head').find('byline')
    for d in head:
        if d.get("name") == "dsk":
            desk = d.get("content")
    if desk == "Editorial Desk":
        icount += 1
        try:
            if auth is not None:
                auth = auth.text
                if auth is not None:
                    acount += 1
                    auth = auth[3:].lower()
                    if ';' in auth:
                        auth = auth.split(';')[0]
                    if ' and ' in auth:
                        auth = auth.split(' and ')
                    if auth in authlist:
                        b = body.find('body.content')
                        if b is not None:
                            for bchild in b:
                                if bchild.get('class') == "full_text":
                                    for p in bchild.findall('p'):
                                        if p.text is not None and len(p.text) > 10:
                                            ptext += p.text

                            if ptext != "":
                                ncount += 1
                                if isinstance(auth, list):
                                    for au in auth:
                                        authord[au].append(ptext)
                                else:
                                    authord[auth].append(ptext)
        except:
            pass
            #print(filename)
            #print(auth)

print("editorials " + str(icount))
print("authors " + str(acount))
print("bodies " + str(ncount))
outroot = "./nyt_texts/2005/"
selectedauth = []
sortedauth = sorted(authord, key=lambda k: len(authord[k]), reverse=True)
for k in sortedauth[:10]:
    selectedauth.append((k, len(authord[k])))
    outfile = outroot + (k.split()[-1]) + ".txt"
    with open(outfile, 'w+') as f:
        f.write(k + '\n')
        for i in authord[k]:
            f.write(i)
    f.close()
for a in selectedauth:
    print(a)
