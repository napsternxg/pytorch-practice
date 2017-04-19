import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Vocab(object):
    def __init__(self, name="vocab",
                 offset_items=tuple([]),
                 UNK=None):
        self.name = name
        self.item2idx = {}
        self.idx2item = []
        self.size = 0
        self.UNK = UNK
        
        self.batch_add(offset_items)
        if UNK is not None:
            self.add(UNK)
            self.UNK_ID = self.item2idx[self.UNK]
        self.offset = self.size
        
    def add(self, item):
        if item not in self.item2idx:
            self.item2idx[item] = self.size
            self.size += 1
            self.idx2item.append(item)
            
    def batch_add(self, items):
        for item in items:
            self.add(item)
            
    def getidx(self, item):
        if item not in self.item2idx:
            if self.UNK is None:
                raise RuntimeError("UNK is not defined. %s not in vocab." % item)
            return self.UNK_ID
        return self.item2idx[item]
            
    def __repr__(self):
        return "Vocab(name={}, size={:d}, UNK={}, offset={:d})".format(
            self.name, self.size,
            self.UNK, self.offset
        )
    
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