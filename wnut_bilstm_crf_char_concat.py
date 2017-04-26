
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use("Agg")
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_utils import *
from pytorch_models import *
from utils import load_sequences, conll_classification_report_to_df
from conlleval import main as conll_eval
import re

sns.set_context("poster")
sns.set_style("ticks")


# In[2]:

TRAIN_CORPUS="data/WNUT_NER/train.tsv"
DEV_CORPUS="data/WNUT_NER/dev.tsv"
TEST_CORPUS="data/WNUT_NER/test.tsv"


# In[3]:

train_corpus = load_sequences(TRAIN_CORPUS, sep="\t", col_ids=(0, -1))
print("Total items in train corpus: %s" % len(train_corpus))
dev_corpus = load_sequences(DEV_CORPUS, sep="\t", col_ids=(0, -1))
print("Total items in dev corpus: %s" % len(dev_corpus))
test_corpus = load_sequences(TEST_CORPUS, sep="\t", col_ids=(0, -1))
print("Total items in test corpus: %s" % len(test_corpus))


# In[5]:
CAP_LETTERS=re.compile(r'[A-Z]')
SMALL_LETTERS=re.compile(r'[a-z]')
NUMBERS=re.compile(r'[0-9]')
PUNCT=re.compile(r'[\.,\"\'!\?;:]')
OTHERS=re.compile(r'[^A-Za-z0-9\.,\"\'!\?;:]')

def get_ortho_feature(word):
    word = CAP_LETTERS.sub("A", word)
    word = SMALL_LETTERS.sub("a", word)
    word = NUMBERS.sub("0", word)
    word = PUNCT.sub(".", word)
    word = OTHERS.sub("%", word)
    return word

def create_vocab(data, vocabs, char_vocab, ortho_word_vocab, ortho_char_vocab, word_idx=0):
    n_vocabs = len(vocabs)
    for sent in data:
        for token_tags in sent:
            for vocab_id in range(n_vocabs):
                vocabs[vocab_id].add(token_tags[vocab_id])
            char_vocab.batch_add(token_tags[word_idx])
            ortho_word = get_ortho_feature(token_tags[word_idx])
            ortho_word_vocab.add(ortho_word)
            ortho_char_vocab.batch_add(ortho_word)
    print("Created vocabs: %s" % (", ".join(
        "{}[{}]".format(vocab.name, vocab.size)
        for vocab in vocabs + [char_vocab, ortho_word_vocab, ortho_char_vocab]
    )))


# In[6]:

word_vocab = Vocab("words", UNK="UNK", lower=True)
char_vocab = Vocab("chars", UNK="<U>", lower=False)
ortho_word_vocab = Vocab("ortho_words", UNK="UNK", lower=True)
ortho_char_vocab = Vocab("ortho_chars", UNK="<U>", lower=False)
ner_vocab = Vocab("ner_tags", lower=False)

create_vocab(train_corpus+dev_corpus+test_corpus, [word_vocab, ner_vocab], char_vocab, ortho_word_vocab, ortho_char_vocab)


# In[7]:

def data2tensors(data, vocabs, char_vocab, ortho_word_vocab, ortho_char_vocab, word_idx=0, column_ids=(0, -1)):
    vocabs = [vocabs[idx] for idx in column_ids]
    n_vocabs = len(vocabs)
    tensors = []
    char_tensors = []
    for sent in data:
        sent_vecs = [[] for i in range(n_vocabs+3)] # Last 3 are for char vecs, ortho_word and ortho_char
        char_vecs = []
        for token_tags in sent:
            vocab_id = 0 # First column is the word
            ortho_word = get_ortho_feature(token_tags[vocab_id])
            # lowercase the word
            sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id].lower())
                )
            for vocab_id in range(1, n_vocabs):
                sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id])
                )
            sent_vecs[-3].append(
                [char_vocab.getidx(c) for c in token_tags[word_idx]]
            )
            sent_vecs[-2].append(
                    ortho_word_vocab.getidx(ortho_word)
                )
            sent_vecs[-1].append(
                [ortho_char_vocab.getidx(c) for c in ortho_word]
            )
        tensors.append(sent_vecs)
    return tensors


# In[8]:

train_tensors = data2tensors(train_corpus, [word_vocab, ner_vocab], char_vocab, ortho_word_vocab, ortho_char_vocab)
dev_tensors = data2tensors(dev_corpus, [word_vocab, ner_vocab], char_vocab, ortho_word_vocab, ortho_char_vocab)
test_tensors = data2tensors(test_corpus, [word_vocab, ner_vocab], char_vocab, ortho_word_vocab, ortho_char_vocab)
print("Train: ({}, {}), Dev: ({}, {}), Test: ({}, {})".format(
    len(train_tensors), len(train_tensors[0]),
    len(dev_tensors), len(dev_tensors[0]),
    len(test_tensors), len(test_tensors[0])
))


# In[9]:

embedding_file="data/WNUT_NER/wnut_vecs.txt"
cache_file="wnut_ner.twitter.400.npy"
ndims=400
pretrained_embeddings = load_word_vectors(embedding_file, ndims, word_vocab, cache_file)


# In[10]:

def plot_losses(train_losses, eval_losses=None, plot_std=False, ax=None):
    if ax is None:
        ax = plt.gca()
    for losses, color, label in zip(
        [train_losses, eval_losses],
        ["0.5", "r"],
        ["Train", "Eval"],
    ):
        mean_loss, std_loss = zip(*losses)
        mean_loss = np.array(mean_loss)
        std_loss = np.array(std_loss)
        ax.plot(
            mean_loss, color=color, label=label,
            linestyle="-", 
        )
        if plot_std:
            ax.fill_between(
                np.arange(mean_loss.shape[0]),
                mean_loss-std_loss,
                mean_loss+std_loss,
                color=color,
                alpha=0.3
            )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Loss ($\pm$ S.D.)")
    
    
def print_predictions(corpus, predictions, filename, label_vocab):
    with open(filename, "w+") as fp:
        for seq, pred in zip(corpus, predictions):
            for (token, true_label), pred_label in zip(seq, pred):
                pred_label = label_vocab.idx2item[pred_label]
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=fp)
            print(file=fp) # Add new line after each sequence


# In[11]:

# ## Class based

# In[19]:

class BiLSTMTaggerWordCRFModel(ModelWrapper):
    def __init__(self, model,
                 loss_function,
                 use_cuda=False, grad_max_norm=5):
        self.model = model
        self.loss_function = None
        self.grad_max_norm=grad_max_norm

        self.use_cuda = use_cuda
        if self.use_cuda:
            #[k.cuda() for k in self.model.modules()]
            self.model.cuda()

    def post_backward(self):
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_max_norm)

    def _process_instance_tensors(self, instance_tensors, volatile=False):
        X, Y, X_char, X_ortho, X_char_ortho = instance_tensors
        X = Variable(torch.LongTensor([X]), requires_grad=False, volatile=volatile)
        X_char = charseq2varlist(X_char, volatile=volatile)
        X_ortho = Variable(torch.LongTensor([X_ortho]), requires_grad=False, volatile=volatile)
        X_char_ortho = charseq2varlist(X_char_ortho, volatile=volatile)
        Y = torch.LongTensor(Y)
        return X, X_char, X_ortho, X_char_ortho, Y

    def get_instance_loss(self, instance_tensors, zero_grad=True):
        if zero_grad:
            ## Clear gradients before every update else memory runs out
            self.model.zero_grad()
        X, X_char, X_ortho, X_char_ortho, Y = instance_tensors
        if self.use_cuda:
            X = X.cuda(async=True)
            X_char = [t.cuda(async=True) for t in X_char]
            X_ortho = X_ortho.cuda(async=True)
            X_char_ortho = [t.cuda(async=True) for t in X_char_ortho]
            Y = Y.cuda(async=True)
        return self.model.loss([(X, X_char), (X_ortho, X_char_ortho)], Y)
        
    def predict(self, instance_tensors):
        X, X_char, X_ortho, X_char_ortho, Y = self._process_instance_tensors(instance_tensors, volatile=True)
        if self.use_cuda:
            X = X.cuda(async=True)
            X_char = [t.cuda(async=True) for t in X_char]
            X_ortho = X_ortho.cuda(async=True)
            X_char_ortho = [t.cuda(async=True) for t in X_char_ortho]
            Y = Y.cuda(async=True)
        emissions = self.model.forward([(X, X_char), (X_ortho, X_char_ortho)])
        return self.model.crf.forward(emissions)[1]


use_cuda=True
hidden_size=128
batch_size=64

char_emb_size=30
output_channels=200
kernel_sizes=[3]

word_emb_size=400
aux_emb_size=100

main_total_emb_dims=700
char_embed_kwargs=dict(
    vocab_size=char_vocab.size,
    embedding_size=char_emb_size,
    out_channels=output_channels,
    kernel_sizes=kernel_sizes
)

word_char_embedding = WordCharEmbedding_tuple(
        word_vocab.size, word_emb_size,
        char_embed_kwargs, dropout=0.5,
        aux_embedding_size=aux_emb_size,
        concat=True)


ortho_char_emb_size=30
output_channels=200
kernel_sizes=[3]
ortho_word_emb_size=200
ortho_total_emb_dims=400

ortho_char_embed_kwargs=dict(
    vocab_size=ortho_char_vocab.size,
    embedding_size=ortho_char_emb_size,
    out_channels=output_channels,
    kernel_sizes=kernel_sizes
)

ortho_word_char_embedding = WordCharEmbedding_tuple(
        ortho_word_vocab.size, ortho_word_emb_size,
        ortho_char_embed_kwargs, dropout=0.5, concat=True)


concat_embeddings = ConcatInputs([word_char_embedding, ortho_word_char_embedding])

# Assign glove embeddings
assign_embeddings(word_char_embedding.word_embeddings, pretrained_embeddings, fix_embedding=True)

n_embed=main_total_emb_dims + ortho_total_emb_dims # Get this using char embedding and word embed and ortho embeddings
model_wrapper = BiLSTMTaggerWordCRFModel(
    BiLSTMTaggerWordCharCRF(concat_embeddings, n_embed, hidden_size, ner_vocab.size),
    None, use_cuda=use_cuda, grad_max_norm=5)


# In[33]:
model_prefix="BiLSTMCharConcatCRF_WNUT_NER_ortho"
n_epochs=50

load_model = True

if load_model:
    model_wrapper.load("{}.pth".format(model_prefix))
    print("Loaded model from {}.pth".format(model_prefix))

training_history = training_wrapper(
    model_wrapper, train_tensors, 
    eval_tensors=dev_tensors,
    optimizer=optim.Adam,
    optimizer_kwargs={
        "lr": 0.1,
        "weight_decay": 1e-2
    },
    n_epochs=n_epochs,
    batch_size=batch_size,
    use_cuda=use_cuda,
    log_file="{}.log".format(model_prefix),
    #early_stopping=0.001,
    save_best=True,
    save_path="{}.pth".format(model_prefix)
)
#model_wrapper.save("{}.pth".format(model_prefix))
model_wrapper.load("{}.pth".format(model_prefix))

# In[34]:

fig, ax = plt.subplots(1,1)
plot_losses(training_history["training_loss"],
            training_history["evaluation_loss"],
            plot_std=True,
            ax=ax)
ax.legend()
sns.despine(offset=5)
plt.savefig("{}.pdf".format(model_prefix))

for title, tensors, corpus in zip(
    ["train", "dev", "test"],
    [train_tensors, dev_tensors, test_tensors],
    [train_corpus, dev_corpus, test_corpus],
                         ):
    predictions = model_wrapper.predict_batch(tensors, title=title)
    print_predictions(corpus, predictions, "%s.wnut.conll" % title, ner_vocab)
    conll_eval(["conlleval", "%s.wnut.conll" % title]) 


