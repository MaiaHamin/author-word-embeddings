import glob
import numpy as np
import os
import random
#from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy import spatial
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA



oneb = np.zeros(50)
oneb[0] = 1
twob = np.zeros(50)
twob[12] = 1
threeb = np.zeros(50)
threeb[25] = 1
fourb = np.zeros(50)
fourb[37] = 1
anchorwords = {'the':oneb, 'and':twob, 'are':threeb, 'is':fourb}



def loadfromfile(filename):
    fileembeds = {}
    with open(filename) as file:
        sepd = file.read().split(']')
        for line in sepd[:-1]:
            line = line.replace('[','')
            comps = line.split()
            embvec = [float(f) for f in comps[1:]]
            fileembeds[comps[0]] = embvec
    return fileembeds

def transform(targembed, testembeds, anchors):
    outembeds = {}
    for key,testembed in testembeds.items():
        postmat = []
        premat = []
        for word in anchors:
            premat.append(targembed[word])
            postmat.append(testembed[word])
        postmat = np.array(postmat)
        A, res, rank, s = np.linalg.lstsq(premat, postmat)
        allmat = []
        allkey = []
        for w,e in testembed.items():
            allmat.append([float(i) for i in e])
            allkey.append(w)
        trans2 = np.dot(np.array(list(allmat)), A)
        transembeds = {}
        for i in range(len(allkey)):
            transembeds[allkey[i]] = trans2[i]
        outembeds[key] = transembeds
    return outembeds

def getsharedwords(embeddict):
    sharedkeys = set()
    emptyset = False
    for files,embeds in embeddict.items():
        for file, embed in embeds.items():
            embedkeys = set(list(embed.keys()))
            if len(sharedkeys) == 0:
                if emptyset:
                    print("No shared words")
                else:
                    sharedkeys = embedkeys
                    emptyset = True
            else:
                sharedkeys = set.intersection(sharedkeys, embedkeys)
    return sharedkeys

def makedicts(trainroot, testroot):
    trains = glob.glob(trainroot + "*.txt")
    tests = glob.glob(testroot + "*.txt")
    traind = {}
    testd = {}
    first = True
    do_transform = False
    targets = {}
    for t in trains:
        embeds = loadfromfile(t)
        if do_transform:
            if first:
                traind[os.path.basename(t)] = embeds
                for word in anchorwords:
                    targets[word] = embeds[word]
            else:
                traind[os.path.basename(t)] = transform(embeds, targets)
        else:
            traind[os.path.basename(t)] = embeds
    for t in tests:
        embeds = loadfromfile(t)
        testd[os.path.basename(t)] = embeds
    return traind, testd

def makedict(tr):
    trains = glob.glob(tr + "*.txt")
    traind = {}
    for t in trains:
        embeds = loadfromfile(t)
        ks = []
        vs = []
        for k, v in embeds.items():
            ks.append(k)
            vs.append(v)
        norm = np.linalg.norm(vs)
        normed = vs / norm
        normedembeds = {}
        for i in range(len(vs)):
            normedembeds[ks[i]] = vs[i]
        traind[os.path.basename(t)] = normedembeds
    return traind


def cumdif(traind, testd):
    cumdifs = {}
    cumcounts = {}
    trainsets = traind.keys()
    for fname, embeds in testd.items():
        filecumdifs = {}
        filecumcounts = {}
        for f in trainsets:
            filecumdifs[f] = 0.
            filecumcounts[f] = 0.
        for word, vec in embeds.items():
            for tset in trainsets:
                newvec = traind[tset].get(word)
                if newvec is not None:
                    filecumdifs[tset] += np.linalg.norm(np.array(vec) - np.array(newvec))
                    filecumcounts[tset] += 1.
        for w,v in filecumdifs.items():
            filecumdifs[w] = v / filecumcounts[w]
        cumdifs[fname] = filecumdifs
    return cumdifs


def get_pres_analogs(analogfile, outfile, embeds):
    outf = open(outfile, "w+")
    count = 0
    with open(analogfile, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(" ")
            print(words)
            aninc = True
            for word in words:
                for fname,embed in embeds.items():
                    if word not in embed:
                        aninc = False
            if aninc:
                outf.write(line)
                count += 1
    print(count)
    return

def rank_cos_dist(trainembeds, testembed):
    rankd = {}
    distd = {}
    testkeys = list(testembed.keys())
    sharedkeys = set(testkeys)
    namedembeds = trainembeds.items()
    for embed in namedembeds:
        embedkeys = set(embed[1].keys())
        sharedkeys = set.intersection(sharedkeys, embedkeys)
    reps = 20
    for key in namedembeds:
        distd[key[0]] = 0
        rankd[key[0]] = 0
    rankd["notfound"] = 0
    #print("number shared keys: " + str(len(sharedkeys)))
    for word in sharedkeys:
        for i in range(reps):
            newword = random.choice(list(sharedkeys))
            mind = 1000000000000000000000
            bestk = "notfound"
            testdist = spatial.distance.cosine(np.array(testembed[word], dtype=np.float64), np.array(testembed[newword], dtype=np.float64))
            for fname,emset in namedembeds:
                try:
                    w1 = np.array(emset[word], dtype=np.float64)
                    w2 = np.array(emset[newword], dtype=np.float64)
                    tdist = spatial.distance.cosine(w1,w2)
                    distd[fname] += np.abs(tdist - testdist)
                    if np.abs(tdist - testdist) < mind:
                        bestk = fname
                        mind = np.abs(tdist - testdist)
                except KeyError:
                    print("unexpected failure")
                    pass
            rankd[bestk] = rankd[bestk] + 1
    rankd["notfound"] = 0
    return distd, rankd


def rank_word_dist(trainembeds, testembed):
    rankd = {}
    distd = {}
    testkeys = list(testembed.keys())
    sharedkeys = set(testkeys)
    namedembeds = trainembeds.items()
    for embed in namedembeds:
        embedkeys = set(embed[1].keys())
        sharedkeys = set.intersection(sharedkeys, embedkeys)
    reps = 20
    for key in namedembeds:
        distd[key[0]] = 0
        rankd[key[0]] = 0
    rankd["notfound"] = 0
    print("number shared keys: " + str(len(sharedkeys)))
    for word in sharedkeys:
        for i in range(reps):
            newword = random.choice(list(sharedkeys))
            mind = 1000000000000000000000
            bestk = "notfound"
            testdist = np.linalg.norm(np.array(testembed[word], dtype=np.float64)-np.array(testembed[newword], dtype=np.float64))
            for fname,emset in namedembeds:
                try:
                    w1 = np.array(emset[word], dtype=np.float64)
                    w2 = np.array(emset[newword], dtype=np.float64)
                    tdist = np.linalg.norm(w1 - w2)
                    distd[fname] += np.abs(tdist - testdist)
                    if np.abs(tdist - testdist) < mind:
                        bestk = fname
                        mind = np.abs(tdist - testdist)
                except KeyError:
                    print("unexpected failure")
                    pass
            rankd[bestk] = rankd[bestk] + 1
    rankd["notfound"] = 0
    return distd, rankd

def rank_trans_dist(trainembeds, testembed):
    rankd = {}
    distd = {}
    testkeys = list(testembed.keys())
    sharedkeys = set(testkeys)
    namedembeds = trainembeds.items()
    for embed in namedembeds:
        embedkeys = set(embed[1].keys())
        sharedkeys = set.intersection(sharedkeys, embedkeys)
    reps = 20
    for key in namedembeds:
        distd[key[0]] = 0
        rankd[key[0]] = 0
    rankd["notfound"] = 0
    #print("number shared keys: " + str(len(sharedkeys)))
    for word in sharedkeys:
        for i in range(reps):
            mind = 1000000000000000000000
            bestk = "notfound"
            for fname,emset in namedembeds:
                try:
                    tdist = np.linalg.norm(testembed[word] - emset[word])
                    distd[fname] += np.abs(tdist)
                    if np.abs(tdist) < mind:
                        bestk = fname
                        mind = np.abs(tdist)
                except KeyError:
                    print("unexpected failure")
                    pass
            rankd[bestk] = rankd[bestk] + 1
    rankd["notfound"] = 0
    return distd, rankd



def getconf(dists):
    runsum = 0
    confd = {}
    for k,v in dists.items():
        runsum += v
    for k,v in dists.items():
        confd[k] = v / runsum
    return confd

def stattest():
    #reinclude 25
    vsizes = [500]
    emsizes = [150]
    tlens = [10000]

    outfile = "./nyt_new/smtolg.txt"
    out = open(outfile, "w+")
    emrankcor = {}
    emdistcor = {}
    vrankcor = {}
    vdistcor = {}
    tsrankcor = {}
    tsdistcor = {}
    for t in tlens:
        for e in emsizes:
            numfiles = 3
            for v in vsizes:
                for i in range(1):
                    if e not in emrankcor:
                        emrankcor[e] = []
                        print(str(e) + " added")
                    if e not in emdistcor:
                        emdistcor[e] = []
                    if v not in vrankcor:
                        vrankcor[v] = []
                    if v not in vdistcor:
                        vdistcor[v] = []
                    if t not in tsrankcor:
                        tsrankcor[t] = []
                    if t not in tsdistcor:
                        tsdistcor[t] = []

                    nfiles = 7
                    root = "./nyt_new/test/" + str(t) + "_" + str(v) + "_" + str(e) + "_" + str(i) + "/"
                    roots = glob.glob(root)
                    if roots != []:
                        dicts = {}
                        for r in roots:
                            dicts[r] = makedict(r)
                        for j in range(1,5):
                            testroot = "./canada_setvocs/embeds/" + str(t) + "_" + str(v) + "_" + str(e) + "_" + str(j) + "/"
                            testroots = glob.glob(testroot)
                            if testroots != []:
                                print(roots)
                                print(testroots)
                                testdicts = {}
                                for r in testroots:
                                    testdicts[r] = makedict(r)

                                sharedwords = set.intersection(getsharedwords(dicts), getsharedwords(testdicts))
                                targetwords = list(sharedwords)[:10]


                                for r1, t1 in dicts.items():
                                    for r2, t2 in testdicts.items():
                                        if r1 != r2:

                                            trialrankcor = 0
                                            trialdistcor = 0

                                            for k, val in t1.items():
                                                #dist, ranks = rank_trans_dist(transform(val, t2, targetwords), val)
                                                dist, ranks = rank_word_dist(t2, val)
                                                sdist = sorted(dist, key=lambda k1: dist[k1])
                                                srank = sorted(ranks, key=lambda k1: ranks[k1], reverse=True)
                                                predr = srank[0]
                                                predd = sdist[0]

                                                if predr[-8:-5] == str(os.path.basename(k))[-8:-5]:
                                                    trialrankcor += 1
                                                if predd[-8:-5] == str(os.path.basename(k))[-8:-5]:
                                                    trialdistcor += 1
                                            emrankcor[e].append(trialrankcor)
                                            tsrankcor[t].append(trialrankcor)
                                            vrankcor[v].append(trialrankcor)
                                            emdistcor[e].append(trialdistcor)
                                            tsdistcor[t].append(trialdistcor)
                                            vdistcor[v].append(trialdistcor)


                        #for k in emtot.keys():
    for kint in emrankcor.keys():
        k = str(kint)
        print("rank accuracy for emb / voc " + k + ":")
        print(float(sum(emrankcor[kint])) / float(nfiles * len(emrankcor[kint])))
        print("dist accuracy for emb / voc " + k + ":")
        print(float(sum(emdistcor[kint])) / float(nfiles * len(emdistcor[kint])))
        out.write(str(k) + "\nrank")
        for acc in emrankcor[kint]:
            out.write(str(acc) + ",")
        out.write(k + "\ndist")
        for acc in emdistcor[kint]:
            out.write(str(acc) + ",")
        out.write("\nsummary:")
        out.write(str(k) + ", " + str(float(sum(emrankcor[kint])) / float(nfiles * len(emrankcor[kint]))) + "," + str(float(sum(emdistcor[kint])) / float(nfiles * len(nfiles * emdistcor[kint]))) + "\n")
    for kint in vrankcor.keys():
        k = str(kint)
        print("rank accuracy for emb / voc " + k + ":")
        print(float(sum(vrankcor[kint])) / float(nfiles * len(vrankcor[kint])))
        print("dist accuracy for emb / voc " + k + ":")
        print(float(sum(vdistcor[kint])) / float(nfiles * len(vdistcor[kint])))
        out.write(str(k) + "\nrank")
        for acc in vrankcor[kint]:
            out.write(str(acc) + ",")
        out.write(str(k) + "\ndist")
        for acc in vdistcor[kint]:
            out.write(str(acc) + ",")
        out.write("\nsummary:")
        out.write(str(k) + ", " + str(float(sum(vrankcor[kint])) / float(nfiles * len(vrankcor[kint]))) + "," + str(float(sum(vdistcor[kint])) / float(nfiles * len(vdistcor[kint]))) + "\n")
    for kint in tsrankcor.keys():
        k = str(kint)

        print("rank accuracy for emb / voc " + k + ":")
        print(float(sum(tsrankcor[kint])) / float(nfiles * len(tsrankcor[kint])))
        print("dist accuracy for emb / voc " + k + ":")
        print(float(sum(tsdistcor[kint])) / float(nfiles * len(tsdistcor[kint])))
        out.write(str(k) + "\nrank")
        for acc in tsrankcor[kint]:
            out.write(str(acc) + ",")
        out.write(str(k) + "\ndist")
        for acc in tsdistcor[kint]:
            out.write(str(acc) + ",")
        out.write("\nsummary:")
        out.write(str(k) + ", " + str(float(sum(tsrankcor[kint])) / float(nfiles * len(tsrankcor[kint]))) + "," + str(float(sum(tsdistcor[kint])) / float(nfiles * len(tsdistcor[kint]))) + "\n")






def printtestfed():
    emsizes = [25, 50, 75]
    dpreds = {}
    rpreds = {}
    counts = {}
    for tf in glob.glob("./federalist/disputed/1000_100_25_0/*.txt"):
        t = os.path.basename(tf)
        rpreds[t] = {"embeds_hamilton.txt":0, "embeds_madison.txt":0,"embeds_both.txt":0}
        dpreds[t] = {"embeds_hamilton.txt":0, "embeds_madison.txt":0,"embeds_both.txt":0}
        counts[t] = 0
    for e in emsizes:
        trainroot = "./federalist/*_" + str(e) + "_0/"
        trainroots = glob.glob(trainroot)
        print(trainroots)
        testroot = "./federalist/disputed/*_" + str(e) + "_0/"
        testroots = glob.glob(testroot)
        traindicts = {}
        testdicts = {}
        for r in trainroots:
            traindicts[r] = makedict(r)
        for r in testroots:
            testdicts[r] = makedict(r)



        print("embed dim: " + str(e))
        for r1, t1 in testdicts.items():
            r1params = str(r1).split("/")[-2].split("_")
            for r2, t2 in traindicts.items():
                r2params = str(r2).split("/")[-2].split("_")
                for k, v in t1.items():
                    dist, ranks = rank_word_dist(t2, v)
                    sdist = sorted(dist, key=lambda k1: dist[k1], reverse=True)
                    srank = sorted(ranks, key=lambda k1: ranks[k1], reverse=True)
                    predr = srank[0]
                    predd = sdist[0]
                    rpreds[k][predr] += 1
                    dpreds[k][predd] += 1
                    counts[k] += 1
    for k in rpreds.keys():
        print("For paper no. " + str(k))
        for auth,val in rpreds[k].items():
            print("rank prediction for " + auth + ":")
            print(float(val) / counts[k])
        for auth,val in dpreds[k].items():
            print("dist preduction for " + auth + ":")
            print(float(val) / counts[k])

def printtestnyt():
    tsizes = [10000, 15000, 20000]
    emsizes = [150]
    vsizes = [500]
    outf = open("./brontevalues.txt", "w+")
    dpreds = {}
    rpreds = {}
    counts = {}
    for t in tsizes:
        for e in emsizes:
            for vs in vsizes:
                for i in range(9):
                    trainroot = "./bronte/embeds_long_test/" + str(t) + "_" + str(vs) + "_" + str(e)+ "_" + str(i) + "/"
                    trainroots = glob.glob(trainroot)
                    print(trainroots)
                    dicts = {}
                    for r in trainroots:
                        dicts[r] = makedict(r)
                    for j in range(1):
                        testroot = "./canada_setvocs/embeds/sanity_check/" + str(t) + "_" + str(vs) + "_" + str(e) + "_2/"
                        testroots = glob.glob(testroot)
                        print(testroots)
                        testdicts = {}
                        for r in testroots:
                            testdicts[r] = makedict(r)

                        sharedwords = set.intersection(getsharedwords(dicts), getsharedwords(testdicts))
                        targetwords = list(sharedwords)[:10]

                        print("embed dim: " + str(e) + ", vocab: " + str(vs))
                        for r1, t1 in dicts.items():
                            for r2, t2 in testdicts.items():
                                for k, v in t1.items():
                                    outf.write(k + ",")
                                    print("projected:")
                                    dist, rank = rank_trans_dist(transform(v, t2, targetwords), v)
                                    sdist = sorted(dist, key=lambda k1: dist[k1])
                                    srank = sorted(rank, key=lambda k1: rank[k1], reverse=True)
                                    print("==========")
                                    outf.write(str(rank[srank[0]] / sum([rank[i] for i in srank])) + ",")
                                    print("--------")
                                    outf.write(str(dist[sdist[0]] / sum([dist[i] for i in sdist])) + ",")
                                    print("=========")
                                    print("distance:")
                                    dist, rank = rank_word_dist(transform(v, t2, targetwords), v)
                                    sdist = sorted(dist, key=lambda k1: dist[k1])
                                    srank = sorted(rank, key=lambda k1: rank[k1], reverse=True)
                                    print("==========")
                                    outf.write(str((rank[srank[0]] / sum([rank[i] for i in srank]))) + ",")
                                    print("--------")
                                    outf.write(str((dist[sdist[0]] / sum([dist[i] for i in sdist]))) + ",")
                                    print("=========")
                                    print("cosine:")
                                    dist, rank = rank_cos_dist(transform(v, t2, targetwords), v)
                                    sdist = sorted(dist, key=lambda k1: dist[k1])
                                    srank = sorted(rank, key=lambda k1: rank[k1], reverse=True)
                                    print("==========")
                                    outf.write(str(rank[srank[0]] / sum([rank[i] for i in srank])) + ",")
                                    print("--------")
                                    outf.write(str(dist[sdist[0]] / sum([dist[i] for i in sdist])) + ",")
                                    print("=========")






def printtestfed():
    emsizes = [50, 100, 150]
    dpreds = {}
    rpreds = {}
    counts = {}
    for tf in glob.glob("./federalist/disputed/1000_100_25_0/*.txt"):
        t = os.path.basename(tf)
        rpreds[t] = {"embeds_hamilton.txt":0, "embeds_madison.txt":0,"embeds_both.txt":0}
        dpreds[t] = {"embeds_hamilton.txt":0, "embeds_madison.txt":0,"embeds_both.txt":0}
        counts[t] = 0
    for e in emsizes:
        trainroot = "./federalist/*_" + str(e) + "_0/"
        trainroots = glob.glob(trainroot)
        print(trainroots)
        testroot = "./federalist/disputed/*_" + str(e) + "_0/"
        testroots = glob.glob(testroot)
        traindicts = {}
        testdicts = {}
        for r in trainroots:
            traindicts[r] = makedict(r)
        for r in testroots:
            testdicts[r] = makedict(r)


        print("embed dim: " + str(e))
        for r1, t1 in testdicts.items():
            r1params = str(r1).split("/")[-2].split("_")
            for r2, t2 in traindicts.items():
                r2params = str(r2).split("/")[-2].split("_")
                for k, v in t1.items():
                    dist, ranks = rank_word_dist(t2, v)
                    sdist = sorted(dist, key=lambda k1: dist[k1], reverse=True)
                    srank = sorted(ranks, key=lambda k1: ranks[k1], reverse=True)
                    predr = srank[0]
                    predd = sdist[0]
                    rpreds[k][predr] += 1
                    dpreds[k][predd] += 1
                    counts[k] += 1
    for k in rpreds.keys():
        print("For paper no. " + str(k))
        for auth,val in rpreds[k].items():
            print("rank prediction for " + auth + ":")
            print(float(val) / counts[k])
        for auth,val in dpreds[k].items():
            print("dist preduction for " + auth + ":")
            print(float(val) / counts[k])

def sanity():
    emsizes = [100]
    dpreds = {}
    rpreds = {}
    counts = {}
    for e in emsizes:
        trainroot = "./sanity/train/5000_500_100/"
        trainroots = glob.glob(trainroot)
        print(trainroots)
        testroot = "./sanity/test/5000_500_100/"
        testroots = glob.glob(testroot)
        print(testroots)
        traindicts = {}
        testdicts = {}
        for r in trainroots:
            traindicts[r] = makedict(r)
        for r in testroots:
            testdicts[r] = makedict(r)

        for r1, t1 in testdicts.items():
            r1params = str(r1).split("/")[-2].split("_")
            for r2, t2 in traindicts.items():
                r2params = str(r2).split("/")[-2].split("_")
                for k, v in t1.items():
                    dist, ranks = rank_cos_dist(t2, v)
                    sdist = sorted(dist, key=lambda k1: dist[k1])
                    srank = sorted(ranks, key=lambda k1: ranks[k1], reverse=True)
                    predr = srank[0]
                    predd = sdist[0]
                    print(k)
                    print(predr + " " + str(ranks[predr]))
                    print(predd + " " + str(dist[predd]))
                    print("----------")

def kmeans():
    vsizes = [500, 1000, 1500]
    emsizes = [50, 100, 150, 200]
    embed_dists = {}
    for e in emsizes:
        for v in vsizes:
            nclasses = 5
            print("cluster for embed dim " + str(e))
            trainroot = "./canadalong/*/*" + str(v) + "_" + str(e) + "/"
            trainroots = glob.glob(trainroot)
            print(trainroots)
            traindicts = {}
            sharedkeys = set()
            for r in trainroots:
                rembeds = makedict(r)
                for ke,re in rembeds.items():
                    k = str(os.path.dirname(r)) + "/" + str(ke)
                    traindicts[k] = re
                    embed_dists[k] = []
            print(traindicts.keys())
            for file,embed in traindicts.items():
                embedkeys = set(list(embed.keys()))
                if len(sharedkeys) == 0:
                    sharedkeys = embedkeys
                    print("once per embed dim")
                else:
                    sharedkeys = set.intersection(sharedkeys, embedkeys)
            print("number shared words: " + str(len(sharedkeys)))
            swordlist = list(sharedkeys)
            nreps = 150
            for i in range(nreps):
                word1 = random.choice(swordlist)
                word2 = random.choice(swordlist)
                for k,em in traindicts.items():
                    embed_dists[k].append(np.linalg.norm(np.array(em[word1], dtype=np.float64)- np.array(em[word2], dtype=np.float64)))


            dists = []
            labels = []
            for k, ds in embed_dists.items():
                labels.append(k)
                dists.append(ds)

            print(np.unique(list(map(len, dists))))
            kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(dists)
            calcedlabels = list(kmeans.labels_)
            labdict = {}
            for i in range(len(calcedlabels)):
                splitlab = labels[i].split("/")
                auth = splitlab[-1]
                params = splitlab[-2]
                if calcedlabels[i] in labdict:
                    if auth in labdict[calcedlabels[i]]:
                        labdict[calcedlabels[i]][auth].append(params)
                    else:
                        labdict[calcedlabels[i]][auth] = [params]
                else:
                    labdict[calcedlabels[i]] = {auth:[params]}


            for k,v in labdict.items():
                print(k)
                for k2,v2 in v.items():
                    print(str(k2) + ": " + str(v2))
                print("=========")

def logreg():
    vsizes = [50]
    tsizes = [2000, 5000, 10000]
    emsizes = [150]
    embed_dists = {}
    test_embed_dists = {}
    for t in tsizes:
        for i in range(1):
        #for v in vsizes:
            print("log reg for t-dim " + str(t))
            trainroot = "./nyt_new/train/" + str(t) + "_500_100/*/"
            trainroots = glob.glob(trainroot)
            print(trainroots)
            if trainroots != []:
                traindicts = {}
                sharedkeys = set()
                authlist = []
                for r in trainroots:
                    rembeds = makedict(r)
                    if authlist == []:
                        authlist = list(set([str(k)[-8:] for k in rembeds.keys()]))
                    for ke,re in rembeds.items():
                        k = str(os.path.dirname(r)) + "/" + str(ke)
                        traindicts[k] = re
                        embed_dists[k] = []

                testroot = "./nyt_new/test/" + str(t) + "_500_100/*/"
                testroots = glob.glob(testroot)
                testdicts = {}
                print(testroots)
                for r in testroots:
                    rembeds = makedict(r)
                    for ke,re in rembeds.items():
                        k = str(os.path.dirname(r)) + "/" + str(ke)
                        testdicts[k] = re
                        test_embed_dists[k] = []
                for file,embed in traindicts.items():
                    embedkeys = set(list(embed.keys()))
                    if len(sharedkeys) == 0:
                        sharedkeys = embedkeys
                        print("once per embed dim")
                    else:
                        sharedkeys = set.intersection(sharedkeys, embedkeys)
                for file,embed in testdicts.items():
                    embedkeys = set(list(embed.keys()))
                    sharedkeys = set.intersection(sharedkeys, embedkeys)
                print("number shared words: " + str(len(sharedkeys)))
                swordlist = list(sharedkeys)
                nreps = 100
                for i in range(nreps):
                    word1 = random.choice(swordlist)
                    word2 = random.choice(swordlist)
                    for k,em in traindicts.items():
                        embed_dists[k].append(np.linalg.norm(np.array(em[word1], dtype=np.float64)- np.array(em[word2], dtype=np.float64)))
                        #embed_dists[k].append(spatial.distance.cosine(np.array(em[word1], dtype=np.float64), np.array(em[word2], dtype=np.float64)))
                    for k, em in testdicts.items():
                        test_embed_dists[k].append(np.linalg.norm(np.array(em[word1], dtype=np.float64)- np.array(em[word2], dtype=np.float64)))
                        #test_embed_dists[k].append(spatial.distance.cosine(np.array(em[word1], dtype=np.float64), np.array(em[word2], dtype=np.float64)))


                dists = []
                labels = []
                for k, ds in embed_dists.items():
                    lab = authlist.index(k.split("/")[-1][-8:])
                    labels+= [lab, lab, lab, lab]
                    dists += [ds, ds, ds, ds]
                pca = PCA(n_components=50)
                transdist = pca.fit_transform(dists)
                testdists = []
                testlabels = []
                for k, ds in test_embed_dists.items():
                    testlabels.append(authlist.index(k.split("/")[-1][-8:]))
                    testdists.append(ds)
                transtest = pca.transform(testdists)

                logreg = LogisticRegression().fit(dists, labels)
                preds = logreg.predict(testdists)
                corcount = 0
                for i in range(len(preds)):
                    #print("pred: " + str(preds[i]) + ", act: " + str(testlabels[i]))
                    if preds[i] == testlabels[i]:
                        corcount += 1
                print("correct: " + str(float(corcount) / float(len(preds))))



def outlierwords():
    #reinclude 25
    vsizes = [500]
    emsizes = [150]
    pdicts = {}
    sharedkeys = set()

    for e in emsizes:
        for v in vsizes:
            for i in range(1):
                nfiles = 5
                root = "./nyt_new/*/*" + str(v) + "_" + str(e) + "/*/"
                roots = glob.glob(root)
                traindicts = {}
                for r in roots:
                    traindicts = makedict(r)
                    pdicts[str(v) + "_" + str(e)] = traindicts
                for file,embed in traindicts.items():
                    embedkeys = set(list(embed.keys()))
                    if len(sharedkeys) == 0:
                        sharedkeys = embedkeys
                        print("once per embed dim")
                    else:
                        sharedkeys = set.intersection(sharedkeys, embedkeys)

    numreps = 100
    print("number shared words: " + str(len(sharedkeys)))
    swordlist = list(sharedkeys)
    print(pdicts.keys())
    auths = {}
    for word1 in swordlist:
        deviations = {}
        for key, traindicts in pdicts.items():
            for j in range(numreps):
                word2 = random.choice(swordlist)
                dlist = []
                klist = []
                for k,em in traindicts.items():
                    klist.append(k)
                    dlist.append(np.linalg.norm(np.array(em[word1], dtype=np.float64)- np.array(em[word2], dtype=np.float64)))
                mean = np.mean(dlist)
                absdif = [np.abs(d - mean) for d in dlist]
                maxind = np.argmax(absdif)
                if klist[maxind] in deviations:
                    deviations[klist[maxind]] += 1
                else:
                    deviations[klist[maxind]] = 1
        sortdev = sorted(deviations, key=lambda k1: deviations[k1], reverse=True)
        if deviations[sortdev[0]] > 2.5 * deviations[sortdev[1]]:
            print(word1 + " || " + str(sortdev[0]) + ": " + str(float(deviations[sortdev[0]] / 100)))











sanity()
