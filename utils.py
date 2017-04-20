import numpy as np
from collections import Counter
import pandas as pd
import re

#import tensorflow as tf

from sklearn.cluster import KMeans


def get_clusters(W_word, n_clusters=10, **kwargs):
    clusterer = KMeans(n_clusters=n_clusters,
            n_jobs=-1, **kwargs)
    cluster_labels = clusterer.fit_predict(W_word)
    return cluster_labels


def read_glove(filename,
               ndims=50):
    vocab = []
    char_vocab = Counter()
    W = []
    with open(filename) as fp:
        for line in fp:
            line = line.rstrip().split()
            word = line[0]
            embed = list(map(float, line[1:]))
            vocab.append(word)
            W.append(embed)
            char_vocab.update(list(word))
    return vocab, char_vocab, np.array(W)


def crf_loss(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    seq_lengths_t = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0),
                tf.int32), axis=-1)
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_true, seq_lengths_t)
    return tf.reduce_mean(-log_likelihood, axis=-1)


def load_sequences(filenames, sep=" ", col_ids=None):
    sequences = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        with open(filename, encoding='utf-8') as fp:
            seq = []
            for line in fp:
                line = line.rstrip()
                if line:
                    line = line.split(sep)
                    if col_ids is not None:
                        line = [line[idx] for idx in col_ids]
                    seq.append(tuple(line))
                else:
                    if seq:
                        sequences.append(seq)
                    seq = []
            if seq:
                sequences.append(seq)
    return sequences


def classification_report_to_df(report):
    report_list = []
    for i, line in enumerate(report.split("\n")):
        if i == 0:
            report_list.append(["class", "precision", "recall", "f1-score", "support"])
        else:
            line = line.strip()
            if line:
                if line.startswith("avg"):
                    line = line.replace("avg / total", "avg/total")
                line = re.split(r'\s+', line)
                line = [line[0]] + list(map(float, line[1:-1])) + [int(line[-1])]
                report_list.append(tuple(line))
    return pd.DataFrame(report_list[1:], columns=report_list[0])  


def conll_classification_report_to_df(report):
    report_list = []
    report_list.append(["class", "accuracy", "precision", "recall", "f1-score", "support"])
    for i, line in enumerate(report.split("\n")):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            continue
        if i == 1:
            line = re.findall(
                'accuracy:\s*([0-9\.]{4,5})%; precision:\s+([0-9\.]{4,5})%; recall:\s+([0-9\.]{4,5})%; FB1:\s+([0-9\.]{4,5})',
                line)[0]
            line = ("overall",) + tuple(map(float, line)) + (0,)
        else:
            line = re.findall(
                '\s*(.+?): precision:\s+([0-9\.]{4,5})%; recall:\s+([0-9\.]{4,5})%; FB1:\s+([0-9\.]{4,5})\s+([0-9]+)',
                line)[0]
            line = (line[0], 0.0) + tuple(map(float, line[1:-1])) + (int(line[-1]),)
        report_list.append(line)
    return pd.DataFrame(report_list[1:], columns=report_list[0])


def get_labels(y_arr):
    return np.expand_dims(
        np.array([
            np.zeros(max_len)
            if y is None else y
            for y in y_arr],
            dtype='int'),
        -1)



def create_tagged_sequence(seq, task2col, default_tag):
    seq_tags = []
    for t in seq:
        try:
            tag = default_tag._replace(token=t[0], **{ti: t[ci] for ti, ci in task2col.items()})
        except:
            print("Error processing tag:", t)
            print("Error in sequence: ", seq)
            raise
        seq_tags.append(tag)
    return seq_tags        


def get_tagged_corpus(corpus, *args):
    max_len = 0
    for seq in corpus:
        if seq:
            max_len = max(len(seq), max_len)
            yield create_tagged_sequence(seq, *args)
    print("Max sequence length in the corpus is: %s" % max_len)

def gen_vocab_counts(corpus, tasks, include_chars=False, token_counts=None):
    task_counts = {k: Counter() for k in tasks}
    if token_counts is None:
        token_counts = Counter()
    max_seq_len = 0
    max_word_len = 0
    if include_chars:
        char_counts = Counter()
    for seq in corpus:
        max_seq_len = max(len(seq), max_seq_len)
        for t in seq:
            token_counts[t.token] += 1
            if include_chars:
                char_counts.update(list(t.token))
                max_word_len = max(len(t.token), max_word_len)
            for k in task_counts:
                v = getattr(t, k)
                if v is not None:
                    task_counts[k][v] += 1
    if include_chars:
        return token_counts, task_counts, max_seq_len, char_counts, max_word_len
    return token_counts, task_counts, max_seq_len

def print_predictions(tagged_seq, predictions, filename, label_id=0, task_id=0):
    from sklearn.metrics import classification_report, accuracy_score
    y_true, y_pred = [], []
    with open(filename, "w+") as fp:
        for seq, pred in zip(tagged_seq, predictions[label_id]):
            for tag, label in zip(seq, pred):
                true_label = tag[task_id+1]
                print(u"%s\t%s\t%s" % (tag[0], true_label, label), file=fp)
                y_true.append(true_label)
                y_pred.append(label)
            print(u"", file=fp) 
    
    report = classification_report(y_true, y_pred)
    print(report)
    print("Accuracy: %s" % accuracy_score(y_true, y_pred))
    return classification_report_to_df(report)



