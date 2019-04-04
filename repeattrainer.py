import zipfile
import collections
import numpy as np

import math
import random
import collections
import os
import string
from sklearn.decomposition import PCA

import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time

data_index = 0
data_holder = 0

class Options(object):
  def __init__(self, datafile, vocabulary_size, rstart, rend):
    self.vocabulary_size = vocabulary_size
    self.vocabulary = self.read_data(datafile, rstart, rend)
    data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary,
                                                              self.vocabulary_size)
    self.train_data = self.subsampling(data_or)
    self.sample_table = self.init_sample_table()

  def read_data(self, filename, rstart, rend):
    global data_holder
    data_holder = 0
    translator = str.maketrans("", "", string.punctuation)
    with open(filename, encoding="ISO-8859-1") as f:
      data = f.read().translate(translator).split()
      data = [x.lower() for x in data if x != 'eoood' and x.isalnum()][rstart:rend]
    return data

  def build_dataset(self,words, n_words):
    """Process raw inputs into a ."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, reversed_dictionary

  def init_sample_table(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ (power + 0.0000000000000000000000000000000001)
    table_size = 1e8
    count = np.round(ratio*table_size)
    sample_table = []
    for idx, x in enumerate(count):
      sample_table += [idx]*int(x)
    return np.array(sample_table)
  def weight_table(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ (power + 0.0000000000000000000000000000000001)
    return np.array(ratio)
  def subsampling(self,data):
    count = [ele[1] for ele in self.count]
    frequency = np.array(count)/(sum(count) + 0.0000000000000000000000001)
    P = dict()
    for idx, x in enumerate(frequency):
      if x != 0:
          y = (math.sqrt(x/0.001)+1)*0.001/x
      else:
          y = (math.sqrt(0.000000000000001/0.001)+1)*0.001/0.0000000000000001
      P[idx] = y
    subsampled_data = list()
    for word in data:
      if random.random()<P[word]:
        subsampled_data.append(word)
    return subsampled_data


  def generate_batch(self, window_size, batch_size, count):
    data = self.train_data
    global data_index
    global data_holder
    span = 2 * window_size + 1
    context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size), dtype=np.int64)
    pos_pair = []

    if data_index + span > len(data):
      data_holder += 1
      data_index = data_holder
      self.process = False
    buffer = data[data_index:data_index + span]
    pos_u = []
    pos_v = []

    for i in range(batch_size):
      data_index += 1
      context[i,:] = buffer[:window_size] + buffer[window_size + 1:]
      labels[i] = buffer[window_size]
      if data_index + span > len(data):
        buffer[:] = data[:span]
        data_holder += 1
        data_index = data_holder
        self.process = False
      else:
        buffer = data[data_index:data_index + span]

      for j in range(span-1):
        pos_u.append(labels[i])
        pos_v.append(context[i,j])
    neg_v = np.random.choice(self.sample_table, size=(batch_size * 2 * window_size, count))
    return np.array(pos_u), np.array(pos_v), neg_v


class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.embedding_dim = embedding_dim
    self.init_emb()
  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)
  def forward(self, u_pos, v_pos, v_neg, batch_size):

    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    score  = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = Func.logsigmoid(score).squeeze()

    neg_embed_v = self.v_embeddings(v_neg)

    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = Func.logsigmoid(-1*neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return -1 * loss.sum()/batch_size
  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()
  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data.cpu().numpy()
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word[idx]
      embed = str(embeds[idx])
      fo.write(word+' '+embed+'\n')


class word2vec:
    def __init__(self, inputfile, vocabulary_size=10000, embedding_dim=500, epoch_num=5, batch_size=16, windows_size=5, neg_sample_num=10, range_start=0, range_end=-1, num_its=10000):
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.rstart = range_start
        self.rend = range_end
        self.op = Options(inputfile, self.vocabulary_size, self.rstart, self.rend)
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num
        self.embeds = {}
        self.inputfile = inputfile
        self.num_its = num_its

    def vis(self, embeddings, testwords):
        class_words = self.op.vocab_words
        for w in testwords:
            for kc, vc in class_words.items():
                if vc == w:
                    print(vc + ": ")
                    print(embeddings[kc])

    def train(self, fname):
        model = skipgram(self.vocabulary_size, self.embedding_dim)
        if torch.cuda.is_available():
          model.cuda()
        optimizer = optim.SGD(model.parameters(),lr=0.2)
        for epoch in range(self.epoch_num):
          start = time.time()
          self.op.process = True
          batch_num = 0
          batch_new = 0

          while batch_num < self.num_its:
            pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))




            if torch.cuda.is_available():
              pos_u = pos_u.cuda()
              pos_v = pos_v.cuda()
              neg_v = neg_v.cuda()

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, neg_v,self.batch_size)

            loss.backward()

            optimizer.step()


            if batch_num%30000 == 0:
              torch.save(model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch,batch_num))

              if batch_num%100 == 0:
                end = time.time()
                word_embeddings = model.input_embeddings()
                #self.vis(word_embeddings, testwords)
                batch_new = batch_num
                start = time.time()
            batch_num = batch_num + 1
        model.save_embedding(fname, self.op.vocab_words)




if __name__ == '__main__':
    mintextlen = 60000
    root = "./canadalong/texts/"
    texts = glob.glob(root + '*.txt')
    #texts = ["./federalist/hamilton.txt", "./federalist/madison.txt", "./federalist/both.txt"]
    print(texts)
    trlens = [7500, 10000, 20000]
    vsizes = [250, 500, 1000]
    emsizes = [50, 100, 150]
    for l in trlens:
        for v in vsizes:
            for e in emsizes:
                if e < (v/3) and v <= (l / 10):
                    directory = "./canada_repeated/embeds/" + str(l) + "_" + str(v) + "_" + str(e)
                    if not os.path.exists(directory):
                        print(str(l) + " " + str(v) + " " + str(e) + " " +  " embedding")
                        print("------------------")
                        os.makedirs(directory)
                        rangestart = random.randint(0, mintextlen - l)
                        rangeend = rangestart + l
                        for f in texts:
                            print(f)
                            fname = directory + "/embeds_" + os.path.basename(f)
                            wc = word2vec(f, vocabulary_size=v, embedding_dim=e,
                            range_start=rangestart, range_end=rangeend)
                            wc.train(fname)
                        print(str(l) + " " + str(v) + " " + str(e) + " " +  " embedded")
                        print("------------------")
