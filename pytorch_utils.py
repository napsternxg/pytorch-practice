import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path


class Vocab(object):
    def __init__(self, name="vocab",
                 offset_items=tuple([]),
                 UNK=None, lower=True):
        self.name = name
        self.item2idx = {}
        self.idx2item = []
        self.size = 0
        self.UNK = UNK
        self.lower=lower
        
        self.batch_add(offset_items, lower=False)
        if UNK is not None:
            self.add(UNK, lower=False)
            self.UNK_ID = self.item2idx[self.UNK]
        self.offset = self.size
        
    def add(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        if item not in self.item2idx:
            self.item2idx[item] = self.size
            self.size += 1
            self.idx2item.append(item)
            
    def batch_add(self, items, lower=True):
        for item in items:
            self.add(item, lower=lower)
            
    def in_vocab(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        return item in self.item2idx
        
    def getidx(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        if item not in self.item2idx:
            if self.UNK is None:
                raise RuntimeError("UNK is not defined. %s not in vocab." % item)
            return self.UNK_ID
        return self.item2idx[item]
            
    def __repr__(self):
        return "Vocab(name={}, size={:d}, UNK={}, offset={:d}, lower={})".format(
            self.name, self.size,
            self.UNK, self.offset,
            self.lower
        )
    
    
def load_word_vectors(vector_file, ndims, vocab, cache_file, override_cache=False):
    W = np.zeros((vocab.size, ndims), dtype="float32")
    # Check for cached file and return vectors
    cache_file = Path(cache_file)
    if cache_file.is_file() and not override_cache:
        W = np.load(cache_file)
        return W
    # Else load vectors from the vector file
    total, found = 0, 0
    with open(vector_file) as fp:
        for i, line in enumerate(fp):
            line = line.rstrip().split()
            if line:
                total += 1
                try:
                    assert len(line) == ndims+1,(
                        "Line[{}] {} vector dims {} doesn't match ndims={}".format(i, line[0], len(line)-1, ndims)
                    )
                except AssertionError as e:
                    print(e)
                    continue
                word = line[0]
                idx = vocab.getidx(word) 
                if idx >= vocab.offset:
                    found += 1
                    vecs = np.array(list(map(float, line[1:])))
                    W[idx, :] += vecs
    # Write to cache file
    print("Found {} [{:.2f}%] vectors from {} vectors in {} with ndims={}".format(
        found, found * 100/vocab.size, total, vector_file, ndims))
    norm_W = np.sqrt((W*W).sum(axis=1, keepdims=True))
    valid_idx = norm_W.squeeze() != 0
    W[valid_idx, :] /= norm_W[valid_idx]
    print("Caching embedding with shape {} to {}".format(W.shape, cache_file.as_posix()))
    np.save(cache_file, W)
    return W    
    
class Seq2Vec(object):
    def __init__(self, vocab):
        self.vocab = vocab
        
    def encode(self, seq):
        vec = []
        for item in seq:
            vec.append(self.vocab.getidx(item))
        return vec
    
    def batch_encode(self, seq_batch):
        vecs = [self.encode(seq) for seq in seq_batch]
        return vecs
        
        
class Seq2OneHot(object):
    def __init__(self, size):
        self.size = size
    
    def encode(self, x, as_variable=False):
        one_hot = torch.zeros(self.size)
        for i in x:
            one_hot[i] += 1
        one_hot = one_hot.view(1, -1)
        if as_variable:
            return Variable(one_hot)
        return one_hot
    
    
def print_log_probs(log_probs, label_vocab, label_true=None):
    for i, label_probs in enumerate(log_probs.data.tolist()):
        prob_string = ", ".join([
            "{}: {:.3f}".format(label_vocab.idx2item[j], val)
            for j, val in enumerate(label_probs)
        ])
        true_string = "?"
        if label_true is not None:
            true_string = label_vocab.idx2item[label_true[i]]
            
        print(prob_string, "True label: ", true_string)    
