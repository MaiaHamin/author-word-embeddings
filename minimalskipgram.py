import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

from collections import Counter
import random


def tokenize_corpus(filename):
    sentence_tokens = []
    with open (filename, 'r') as f:
        for line in f:
            sentences = nltk.sent_tokenize(line)
            for s in sentences:
                sentence_words = []
                words = nltk.word_tokenize(s)
                for w in words:
                    sentence_words.append(w)
                sentence_tokens.append(sentence_words)


    return sentence_tokens

tokenized = tokenize_corpus("mobydick.txt")[:50]

vocab = []
word_counts = {}

for sentence in tokenized:
    for token in sentence:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1


threshold = 1e-5
word_size = len(word_counts)
freqs = {word: count/word_size for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
train_words = [[word for word in sentence if random.random() > (1 - p_drop[word])] for sentence in tokenized]

print(train_words[0])
print(train_words[1])

for sentence in train_words:
    for token in sentence:
        if token not in vocab:
            vocab.append(token)

word2idx = {w: i for (i, w) in enumerate(vocab)}
idx2word = {i: w for (i, w) in enumerate(vocab)}

vocab_size = len(vocab)
print("vocab size: " + str(vocab_size))

window_size = 2
idx_pairs = []
for sentence in train_words:
    indices = [word2idx[word] for word in sentence]
    for center_word_pos in range(len(indices)):
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs)

def get_input_layer(word_idx):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 10
W1 = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)

num_epochs = 1000
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

v = W1
u = W2
for i in range(10):
    print(vocab[i])
    print(v[i])
    print(u[i])
    print((u[i] + v[i]) / 2)
