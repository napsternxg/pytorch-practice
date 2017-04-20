import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp_torch(vecs, axis=None):
    ## Use help from: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    if axis < 0:
        axis = vecs.ndimension()+axis
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.expand_as(vecs)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    #print(max_val, out_val)
    return max_val + out_val



class BoWModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(BoWModule, self).__init__()
        self.W = nn.Linear(input_size, output_size)
        
    def forward(self, X):
        return F.log_softmax(self.W(X))


class BoEmbeddingsModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size):
        super(BoEmbeddingsModule, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.W = nn.Linear(embedding_size, output_size)
        
    def forward(self, X):
        hidden_layer = self.word_embeddings(X).mean(1).view(-1,self.word_embeddings.embedding_dim)
        return F.log_softmax(self.W(hidden_layer))
    

    
class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, X):
        seq_embed = self.word_embeddings(X).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        output = self.output(out[-1, :, :])
        return F.log_softmax(output)    

    
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(LSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, X):
        seq_embed = self.word_embeddings(X).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        # Reshape the output to be a tensor of shape seq_len*label_size
        output = self.output(out.view(X.data.size(1), -1))
        return F.log_softmax(output)
    
    
class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 out_channels, kernel_sizes, dropout=0.5):
        super(CharEmbedding, self).__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        # Usage of nn.ModuleList is important
        ## See: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/6
        self.convs1 = nn.ModuleList([nn.Conv2d(1, out_channels, (K, embedding_size), padding=(K-1, 0)) 
                       for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        x = self.char_embeddings(X)
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(self.dropout(i), i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return self.dropout(x)
    
    
class WordCharEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, char_embed_kwargs, dropout=0.5):
        super(WordCharEmbedding, self).__init__()
        self.char_embeddings = CharEmbedding(**char_embed_kwargs)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, X_char=None):
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        word_vecs = self.word_embeddings(X)
        if X_char is not None:
            char_vecs = torch.cat([
                self.char_embeddings(x).unsqueeze(0)
                for x in X_char
            ], 1)
            word_vecs = char_vecs + word_vecs
        return self.dropout(word_vecs)
    
class LSTMTaggerWordChar(nn.Module):
    def __init__(self, word_char_embedding, embedding_size, hidden_size, output_size):
        super(LSTMTaggerWordChar, self).__init__()
        self.word_embeddings = word_char_embedding
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, X, X_char):
        seq_embed = self.word_embeddings(X, X_char).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        # Reshape the output to be a tensor of shape seq_len*label_size
        output = self.output(out.view(X.data.size(1), -1))
        return F.log_softmax(output)
    
    
    
    
class CRFLayer(nn.Module):
    def __init__(self, num_labels):
        super(CRFLayer, self).__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        
    def _forward_alg(self, emissions):
        scores = emissions[0]
        # Get the log sum exp score
        transitions = self.transitions.transpose(-1,-2)
        for i in range(1, emissions.size(0)):
            scores = emissions[i] + log_sum_exp_torch(
                scores.expand_as(transitions) + transitions,
                axis=1)
        return log_sum_exp_torch(scores, axis=-1)
        
    def _score_sentence(self, emissions, tags):
        score = emissions[0][tags[0]]
        if emissions.size()[0] < 2:
            return score
        for i, emission in enumerate(emissions[1:]):
            score = score + self.transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
        return score
    
    def _viterbi_decode(self, emissions):
        emissions = emissions.data.cpu()
        scores = torch.zeros(emissions.size(1))
        back_pointers = torch.zeros(emissions.size()).int()
        scores = scores + emissions[0]
        transitions = self.transitions.data.cpu()
        # Generate most likely scores and paths for each step in sequence
        for i in range(1, emissions.size(0)):
            scores_with_transitions = scores.unsqueeze(1).expand_as(transitions) + transitions
            max_scores, back_pointers[i] = torch.max(scores_with_transitions, 0)
            scores = emissions[i] + max_scores
        # Generate the most likely path
        viterbi = [scores.numpy().argmax()]
        back_pointers = back_pointers.numpy()
        for bp in reversed(back_pointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = scores.numpy().max()
        return viterbi_score, viterbi
        
    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
        
    def forward(self, feats):
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
    
    
class BiLSTMTaggerWordCRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(BiLSTMTaggerWordCRF, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.crf = CRFLayer(output_size)
        
    def forward(self, X):
        seq_embed = self.word_embeddings(X).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        # Reshape the output to be a tensor of shape seq_len*label_size
        output = self.output(out.view(X.data.size(1), -1))
        return output
    
    def loss(self, X, Y):
        feats = self.forward(X)
        return self.crf.neg_log_likelihood(feats, Y)
    
    
class LSTMTaggerWordCharCRF(nn.Module):
    def __init__(self, word_char_embedding, embedding_size, hidden_size, output_size):
        super(LSTMTaggerWordCharCRF, self).__init__()
        self.word_embeddings = word_char_embedding
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.crf = CRFLayer(output_size)
        
    def forward(self, X, X_char):
        seq_embed = self.word_embeddings(X, X_char).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        # Reshape the output to be a tensor of shape seq_len*label_size
        output = self.output(out.view(X.data.size(1), -1))
        return output
    
    def loss(self, X, X_char, Y):
        feats = self.forward(X, X_char)
        return self.crf.neg_log_likelihood(feats, Y)
    