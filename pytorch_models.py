import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm


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


def charseq2varlist(X_chars, volatile=False):
    return [Variable(torch.LongTensor([x]).pin_memory(), requires_grad=False, volatile=volatile) for x in X_chars]


def assign_embeddings(embedding_module, pretrained_embeddings, fix_embedding=False):
    embedding_module.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
    if fix_embedding:
        embedding_module.weight.requires_grad = False


class ModelWrapper(object):
    def __init__(self, model,
                 loss_function,
                 use_cuda=False
                ):
        self.model = model
        self.loss_function = loss_function

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
            
    def batch_process_tensors(self, data_tensors):
        for instance_tensors in data_tensors:
            yield self._process_instance_tensors(instance_tensors)
        
    def _process_instance_tensors(self, instance_tensors, volatile=False):
        raise NotImplementedError("Please define this function explicitly")
        
    def zero_grad(self):
        self.model.zero_grad()

    def post_backward(self):
        ## Implement things like grad clipping or grad norm
        pass
        
    def get_parameters(self):
        return self.model.paramerters()
    
    def set_model_mode(self, training_mode=True):
        if training_mode:
            self.model.train()
        else:
            self.model.eval()
            
    def save(self, filename, verbose=True):
        torch.save(self.model, filename)
        if verbose:
            print("{} model saved to {}".format(self.model.__class__, filename))
        
    def load(self, filename):
        self.model = torch.load(filename)
        if self.use_cuda:
            self.model.cuda()

    def get_instance_loss(self, instance_tensors, zero_grad=True):
        if zero_grad:
        ## Clear gradients before every update else memory runs out
            self.zero_grad()
        raise NotImplementedError("Please define this function explicitly")
        
    def predict(self, instance_tensors):
        raise NotImplementedError("Please define this function explicitly")
        
    def predict_batch(self, batch_tensors, title="train"):
        self.model.eval() # Set model to eval mode
        predictions = []
        for instance_tensors in tqdm(batch_tensors,
                desc="%s predict" % title, unit="instance"):
            predictions.append(self.predict(instance_tensors))
        return predictions
        
        
def get_epoch_function(model_wrapper, optimizer,
                       use_cuda=False):
    def perform_epoch(data_tensors, training_mode=True, batch_size=1, pbar=None):
        model_wrapper.set_model_mode(training_mode)
        step_losses = []
        len_data_tensors = len(data_tensors)
        data_tensor_idxs = np.random.permutation(np.arange(len_data_tensors, dtype="int"))
        n_splits = data_tensor_idxs.shape[0]//batch_size
        title = "train" if training_mode else "eval"
        for batch_tensors_idxs in np.array_split(data_tensor_idxs, n_splits):
            #from IPython.core.debugger import Tracer; Tracer()()
            optimizer.zero_grad()
            #loss = Variable(torch.FloatTensor([0.]))
            losses = []
            for instance_tensors_idx in batch_tensors_idxs:
                instance_tensors = data_tensors[instance_tensors_idx]
                loss = model_wrapper.get_instance_loss(instance_tensors, zero_grad=False)
                losses.append(loss)
                if pbar is not None:
                    pbar.update(1)
            loss = torch.mean(torch.cat(losses))
            #loss = loss/batch_tensors_idxs.shape[0] # Mean loss
            step_losses.append(loss.data[0])
            if training_mode:
                ## Get gradients of model params wrt. loss
                loss.backward()
                ## Model grad specific steps like clipping or norm
                model_wrapper.post_backward()
                ## Optimize the loss by one step
                optimizer.step()
        return step_losses
    return perform_epoch

def write_losses(losses, fp, title="train", epoch=0):
    for i, loss in enumerate(losses):
        print("{:<10} epoch={:<3} batch={:<5} loss={:<10}".format(
            title, epoch, i, loss
        ), file=fp)
    print("{:<10} epoch={:<3} {:<11} mean={:<10.3f} std={:<10.3f}".format(
        title, epoch, "overall", np.mean(losses), np.std(losses)
    ), file=fp)


def training_wrapper(
    model_wrapper, data_tensors,
    eval_tensors=None,
    optimizer=optim.SGD,
    optimizer_kwargs=None,
    n_epochs=10,
    batch_size=1,
    use_cuda=False,
    log_file="training_output.log",
    early_stopping=None,
    save_best=False,
    save_path="best_model.pth",
    reduce_lr_every=5,
    lr_reduce_factor=0.5
):
    """Wrapper to train the model
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    # Fileter out parameters which don't require a gradient
    parameters = filter(lambda p: p.requires_grad, model_wrapper.model.parameters())
    optimizer=optimizer(parameters, **optimizer_kwargs)
    # Start training
    losses = []
    eval_losses = []
    ## Covert data tensors to torch tensors
    data_tensors = list(
        tqdm(
            model_wrapper.batch_process_tensors(data_tensors),
            total=len(data_tensors),
            desc="Proc. train tensors",
            #leave=False,
        )
    )
    if eval_tensors is not None:
        eval_tensors = list(
            tqdm(
                model_wrapper.batch_process_tensors(eval_tensors),
                total=len(eval_tensors),
                desc="Proc. eval tensors",
                #leave=False,
            )
        )
    ## 
    #data_tensors = np.array(data_tensors)
    #if eval_tensors is not None:
    #    eval_tensors = np.array(eval_tensors)
    perform_epoch = get_epoch_function(
        model_wrapper,
        optimizer,
        use_cuda=use_cuda)
    with open(log_file, "w+") as fp:
        with tqdm(total=n_epochs, desc="Epochs", unit="epochs") as epoch_progress_bar:
            for epoch in range(n_epochs):
                with tqdm(
                    total=len(data_tensors),
                    desc="Train", unit="instance", leave=False
                    ) as train_progress_bar:
                    step_losses = perform_epoch(data_tensors, batch_size=batch_size, pbar=train_progress_bar)
                    mean_loss, std_loss = np.mean(step_losses), np.std(step_losses)
                    losses.append((mean_loss, std_loss))
                    write_losses(step_losses, fp, title="train", epoch=epoch)
                if eval_tensors is not None:
                    with tqdm(
                        total=len(eval_tensors),
                        desc="Eval", unit="instance", leave=False) as eval_progress_bar:
                        step_losses = perform_epoch(eval_tensors, training_mode=False, pbar=eval_progress_bar)
                        mean_loss, std_loss = np.mean(step_losses), np.std(step_losses)
                        eval_losses.append((mean_loss, std_loss))
                        write_losses(step_losses, fp, title="eval", epoch=epoch)
                epoch_progress_bar.update(1)
                if early_stopping is not None and epoch > 1:
                    assert isinstance(early_stopping, float), "early_stopping should be either None or float value. Got {}".format(early_stopping)
                    eval_loss_diff = np.abs(eval_losses[-2][0] - eval_losses[-1][0])
                    if eval_loss_diff < early_stopping:
                        epoch_progress_bar.write("Evaluation loss stopped decreased less than {}. Early stopping at epoch {}.".format(early_stopping, epoch))
                        break
                if save_best and save_path is not None:
                    if epoch == 0:
                        best_eval_loss = eval_losses[-1][0]
                        best_epoch = epoch
                        model_wrapper.save(save_path, verbose=False)
                        continue
                    # Save the best model
                    if eval_losses[-1][0] < best_eval_loss:
                        best_eval_loss = eval_losses[-1][0]
                        best_epoch = epoch
                        model_wrapper.save(save_path, verbose=False)
                    if epoch == n_epochs -1:
                        epoch_progress_bar.write("Best model from {} epoch with {:3f} loss".format(best_epoch, best_eval_loss))

                if reduce_lr_every > 0 and lr_reduce_factor > 0 and ((epoch + 1) % reduce_lr_every) == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr']*lr_reduce_factor


    return {
        "training_loss": losses,
        "evaluation_loss": eval_losses
    }




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
        x = self.dropout(x)
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return self.dropout(x)
    
    
class WordCharEmbedding(nn.Module):
    def __init__(self,
            vocab_size, embedding_size,
            char_embed_kwargs, dropout=0.5,
            aux_embedding_size=None,
            concat=False
            ):
        super(WordCharEmbedding, self).__init__()
        self.char_embeddings = CharEmbedding(**char_embed_kwargs)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        if concat and aux_embedding_size is not None:
            ## Only allow aux embedding in concat mode
            self.aux_word_embeddings = nn.Embedding(vocab_size, aux_embedding_size)
        self.concat = concat
        
    def forward(self, X, X_char=None):
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        word_vecs = self.word_embeddings(X)
        if X_char is not None:
            char_vecs = torch.cat([
                self.char_embeddings(x).unsqueeze(0)
                for x in X_char
            ], 1)
            if self.concat:
                embedding_list = [char_vecs, word_vecs]
                if hasattr(self, "aux_word_embeddings"):
                    aux_vecs = self.aux_word_embeddings(X)
                    embedding_list.append(aux_vecs)
                word_vecs = torch.cat(embedding_list, 2)
            else:
                word_vecs = char_vecs + word_vecs
        return self.dropout(word_vecs)

class WordCharEmbedding_tuple(nn.Module):
    def __init__(self,
            vocab_size, embedding_size,
            char_embed_kwargs, dropout=0.5,
            aux_embedding_size=None,
            concat=False
            ):
        super(WordCharEmbedding_tuple, self).__init__()
        self.char_embeddings = CharEmbedding(**char_embed_kwargs)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.concat = concat
        if concat and aux_embedding_size is not None:
            ## Only allow aux embedding in concat mode
            self.aux_word_embeddings = nn.Embedding(vocab_size, aux_embedding_size)
        
    def forward(self, X):
        if isinstance(X, tuple):
            X, X_char = X
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        word_vecs = self.word_embeddings(X)
        if X_char is not None:
            char_vecs = torch.cat([
                self.char_embeddings(x).unsqueeze(0)
                for x in X_char
            ], 1)
            if self.concat:
                embedding_list = [char_vecs, word_vecs]
                if hasattr(self, "aux_word_embeddings"):
                    aux_vecs = self.aux_word_embeddings(X)
                    embedding_list.append(aux_vecs)
                word_vecs = torch.cat(embedding_list, 2)
            else:
                word_vecs = char_vecs + word_vecs
        return self.dropout(word_vecs)

class ConcatInputs(nn.Module):
    def __init__(self, input_modules, dim=2):
        super(ConcatInputs, self).__init__()
        assert isinstance(input_modules, list), "Modules should be a list of input modules"
        self.input_modules = nn.ModuleList(input_modules)
        self.dim = dim

    def forward(self, X):
        assert isinstance(X, list), "X should be a list of input variables"
        concat_vecs = torch.cat([self.input_modules[i](x) for i,x in enumerate(X)], self.dim)
        return concat_vecs


    
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
    
class BiLSTMTaggerWordCharCRF(nn.Module):
    def __init__(self, input_embedding, embedding_size, hidden_size, output_size):
        super(BiLSTMTaggerWordCharCRF, self).__init__()
        self.input_embedding = input_embedding
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.crf = CRFLayer(output_size)
        
    def forward(self, X):
        seq_embed = self.input_embedding(X).permute(1, 0, 2)
        out, hidden = self.lstm(seq_embed)
        # Reshape the output to be a tensor of shape seq_len*label_size
        output = self.output(out.view(out.data.size(0), -1))
        return output
    
    def loss(self, X, Y):
        feats = self.forward(X)
        return self.crf.neg_log_likelihood(feats, Y)
    
