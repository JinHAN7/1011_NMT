#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Beam search  
# - self-attention  
# - EOS mask loss
# //////
# - CNN - done
# - plot - done
# - BLEU - done
# -  Evaluation - done
# - early stopping - done
# -  GPU - done
# - Batch encoder - done
# - pretrained wordvec/dataloader- done
# - sort in collate_func - done
# - save best model - done
# 
# 
# 
# To run this notebook for actual analysis:  
# remove breaker in train_model  
# increase hidden size  
# increase eva_every  
# 

# In[60]:


from __future__ import unicode_literals, print_function, division
import pickle as pkl
import os
from io import open
import unicodedata
import string
import re
import random
import torch
import sacrebleu
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.switch_backend('agg')
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import math
from sacrebleu import raw_corpus_bleu, corpus_bleu

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


device


# ## Data Pre-processing

# In[9]:


data_dir = '/scratch/rw2268/NMT/nlp_data'
SOS_token = 0
EOS_token = 1
PAD_IDX = 2
UNK_IDX = 3
VOCAB_SIZE = 100000


# In[4]:


def pkl_loader(file_name):
    with open(file_name+'.p', 'rb') as f:
        objct = pkl.load(f)
    return(objct)

def pkl_dumper(obj, file_name):
    with open(file_name+'.p', 'wb') as f:
        pkl.dump(obj, f, protocol=None)


# In[12]:


def load_pretrained_wordvec(lan):
    if lan == 'zh':
        filename = 'wiki.zh.vec'
    elif lan == 'en':
        filename = 'wiki-news-300d-1M.vec'
    else:
        filename = 'wiki-news-300d-1M.vec' #######
    with open(os.path.join(data_dir, filename),encoding='utf-8') as f:
        word_vecs = np.zeros((VOCAB_SIZE+4, 300))
        word_vecs[UNK_IDX] = np.random.normal(scale=0.6, size=(300, ))
        word_vecs[SOS_token] = np.random.normal(scale=0.6, size=(300, ))
        word_vecs[EOS_token] = np.random.normal(scale=0.6, size=(300, ))

        words_ft = {'<pad>': PAD_IDX,
                   '<unk>': UNK_IDX, 
                   '<sos>': SOS_token,
                   '<eos>': EOS_token}
        idx2words_ft = {PAD_IDX:'<pad>', UNK_IDX: '<unk>', SOS_token: '<sos>', EOS_token: '<eos>'}
        ordered_words_ft = ['<sos>', '<eos>', '<pad>', '<unk>']
        count = 0
        for i, line in enumerate(f):
            if i == 0:
                continue
            if len(idx2words_ft) >= VOCAB_SIZE: 
                break
            s = line.split()
            if (np.asarray(s[1:]).size != 300):
                print(lan, i, np.asarray(s[1:]).size)
                continue
            word_vecs[count+4, :] = np.asarray(s[1:])
            words_ft[s[0]] = count+4
            idx2words_ft[count+4] = s[0]
            ordered_words_ft.append(s[0])
            count += 1
    word_vecs = torch.FloatTensor(word_vecs)
    pkl_dumper(word_vecs, os.path.join(data_dir, lan + '_word_vecs'))
    pkl_dumper(words_ft, os.path.join(data_dir, lan + '_words_ft'))
    pkl_dumper(idx2words_ft, os.path.join(data_dir, lan + '_idx2words_ft'))
    pkl_dumper(ordered_words_ft, os.path.join(data_dir, lan + '_ordered_words_ft'))


# In[13]:


load_pretrained_wordvec('zh')


# In[14]:


load_pretrained_wordvec('en')


# In[15]:


word_vecs = {}
word2index = {}
index2word = {}
word_vecs['en'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/en_word_vecs')
word_vecs['zh'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/zh_word_vecs')
word2index['en'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/en_words_ft')
word2index['zh'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/zh_words_ft')
index2word['en'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/en_idx2words_ft')
index2word['zh'] = pkl_loader('/scratch/rw2268/NMT/nlp_data/zh_idx2words_ft')
# en_ordered_words_ft = pkl_loader('data/en_ordered_words_ft')
# zh_ordered_words_ft = pkl_loader('data/zh_ordered_words_ft')
VOCAB_SIZE = len(word2index['en'])


# In[16]:


len(word2index['en']), len(word2index['zh'])


# In[17]:


# #word2index['en']['apple'], index2word['en'][10397]
#  word2index['zh']['知识'], index2word['zh'][word2index['zh']['知识']]
# # index2word['en'][0]


# In[18]:


class Lang:
    def __init__(self, name, index2word, word2index):
        self.name = name
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = len(index2word)


# In[19]:


import html
def normalizeString(s):
#     s = re.sub(r"&apos;m", r"am", s)
#     s = re.sub(r"&apos;t", r"not", s)
#     s = re.sub(r"&apos;s", r"is", s)
#     s = re.sub(r"&apos;re", r"are", s)
#     s = re.sub(r"&quot;", r"", s)
#     s = re.sub(r"&apos;ve", r"have", s)
#     s = re.sub(r"&apos;d", r"had", s)
#     s = re.sub(r"&apos;", r"", s)

    s = re.sub(r"([.!?])", r" \1", s)
    s = html.unescape(s)
    return s


# In[20]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


# In[23]:


def loadingLangs(sourcelang, targetlang, setname):
    input_ls = []
    output_ls = []
    print('Reading lines...')
    # Read the file 
    with open(data_dir+'/iwslt-%s-%s/%s.tok.%s'%(sourcelang, targetlang, setname,sourcelang), encoding='utf-8') as f:
        for line in f.readlines():
            input_ls.append([normalizeString(word) for word in line.split()])
    with open(data_dir+'/iwslt-%s-%s/%s.tok.%s'%(sourcelang, targetlang, setname,targetlang), encoding='utf-8') as f:
        for line in f.readlines():
            output_ls.append([normalizeString(word) for word in line.split()])
    pairs = list(zip(input_ls, output_ls))
    pairs = [pair for pair in pairs if (len(pair[0])+len(pair[1]))>0]
    print('Read %s sentence pairs'%(len(input_ls)))
    if sourcelang == 'zh':
        input_lang = Lang(sourcelang, index2word['zh'], word2index['zh'])
    else:
        input_lang = Lang(sourcelang, index2word['zh'], word2index['zh']) ####
    output_lang = Lang(targetlang, index2word['en'], word2index['en'])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# In[25]:


source_tra, target_tra, pairs_tra = loadingLangs('zh', 'en', 'train')
source_val, target_val, pairs_val = loadingLangs('zh', 'en', 'dev')
source_tes, target_tes, pairs_tes = loadingLangs('zh', 'en', 'test')


# In[26]:


print("90% of chinese sentences length = {0}".format(np.percentile([len(x[0]) for x in pairs_tra], 90)))
print("90% of english sentences length = {0}".format(np.percentile([len(x[1]) for x in pairs_tra], 90)))
print(random.choice(pairs_tra))


# ## Dataset

# In[27]:


MAX_SENT_LEN = 42
BATCH_SIZE = 32


# In[28]:


target_tra.word2index['<unk>']


# In[29]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK_IDX for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,source,target):
    input_lang = source
    output_lang = target
    input_tensor = tensorFromSentence(input_lang, pair[0]).reshape((-1))
    target_tensor = tensorFromSentence(output_lang, pair[1]).reshape((-1))
    return (input_tensor, input_tensor.shape[0], target_tensor, target_tensor.shape[0])


# In[30]:


class NMTDataset(Dataset):
    def __init__(self, source, target, pairs):
        self.source = source
        self.target = target
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        inp_ten, inp_len, tar_ten, tar_len = tensorsFromPair(self.pairs[key], self.source, self.target)
        item = {}
        item['inputtensor'] = inp_ten[:MAX_SENT_LEN]
        item['inputlen'] = min(inp_len, MAX_SENT_LEN)
        item['targettensor'] = tar_ten[:MAX_SENT_LEN]
        item['targetlen'] = min(tar_len, MAX_SENT_LEN)
        return item


# In[31]:


train_data = NMTDataset(source_tra, target_tra, pairs_tra)
val_data = NMTDataset(source_tra, target_tra, pairs_val)
test_data = NMTDataset(source_tra, target_tra, pairs_tes)


# ## Dataloader

# In[32]:


#collate function

def collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    src_data, tar_data, src_len, tar_len = [], [], [], []
    for datum in batch:        
        src_datum = np.pad(np.array(datum['inputtensor']),
                                pad_width=((0,MAX_SENT_LEN-datum['inputlen'])),
                                mode="constant", constant_values=PAD_IDX)
        tar_datum = np.pad(np.array(datum['targettensor']),
                                pad_width=((0,MAX_SENT_LEN-datum['targetlen'])),
                                mode="constant", constant_values=PAD_IDX)
        src_data.append(src_datum)
        tar_data.append(tar_datum)
        src_len.append(datum['inputlen'])
        tar_len.append(datum['targetlen'])
        
    ind_dec_order = np.argsort(src_len)[::-1]
    src_data = np.array(src_data)[ind_dec_order]
    src_len = np.array(src_len)[ind_dec_order]
    tar_data = np.array(tar_data)[ind_dec_order]
    tar_len = np.array(tar_len)[ind_dec_order]
    return [torch.from_numpy(np.array(src_data)).to(device),torch.from_numpy(np.array(tar_data)).to(device),
                torch.LongTensor(np.array(src_len)).to(device),torch.LongTensor(np.array(tar_len)).to(device)]
    


# In[33]:


train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_func)
val_loader = torch.utils.data.DataLoader(val_data,
                                           batch_size=BATCH_SIZE,shuffle=False, collate_fn=collate_func)


# In[34]:


# sample data loader
count = 0
for data in train_loader:
    count+=1
    print('input sentence batch: ')
    print(data[0])
    print('input batch dimension: {}'.format(data[0].size()))
    print('target sentence batch: ')
    print(data[1])
    print('target batch dimension: {}'.format(data[1].size()))
    print('input sentence len: ')
    print(data[2])
    print('target sentence len: ')
    print(data[3])
    if count == 1:
        break


# In[36]:


# data[0][1]


# In[37]:


# [source_tra.index2word[i.item()] for i in data[0][1]]


# In[38]:


# [target_tra.index2word[i.item()] for i in data[1][1]]


# ----------------------

# ## Model

# In[39]:


hyper = {
    'HIDDEN_SIZE': 256,
    'LR': 0.0001,
    'EVA_EVERY': 200,
    'DROP_OUT': 0,
    'TEACHER_RATIO': 0.9,
    'N_LAYERS': 2,
    'KER_SIZE': 3,
    'NUM_EPOCHS': 10   
}


# In[40]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True) 
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
    def initHidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device) 
    
    def forward(self, input, hidden):
#         print("input size is: ", input.size())
        batch_size = input.size()[1]
        seq_len = input.size()[0]
        embedded = self.embedding(input).view(seq_len, batch_size, -1)
#         print(seq_len, batch_size)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
#         print("output size is: ", output.size())
        output = self.fc1(output)
#         print("output size is: ", output.size())
        return output, hidden


# In[41]:


class EncoderCNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_vecs=None, ker_size=3, dropout_p=0.5):
        super(EncoderCNN, self).__init__()
        self.hidden_size = hidden_size
        if len(word_vecs) > 0:
            self.embedding = nn.Embedding(input_size, 300, padding_idx=PAD_IDX)
            self.embedding.weight = nn.Parameter(word_vecs)
            self.embedding.requires_grad = False
        else:
            self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
          
          
        self.seq = nn.Sequential(nn.Conv1d(300, hidden_size, kernel_size=ker_size, padding=(ker_size-1)//2),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=ker_size, stride=1, padding=(ker_size-1)//2))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        input = input.transpose(0,1)
        batch_size, seq_len = input.size()
        # input size for conv1d is , N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        output = self.embedding(input).view(batch_size, -1, seq_len)
        output = self.seq(output)
        output = output.view(seq_len, batch_size, -1) 
        output = self.fc(output)
        output = F.relu(self.dropout(output))
        output = self.fc(output)
        hidden = torch.nn.functional.max_pool1d(output.view(batch_size, -1, seq_len), seq_len).permute(2, 0, 1)
        return output, hidden
    
    def initHidden(self,batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# In[42]:


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_SENT_LEN, n_layers=2):

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.view(1,-1)
        seq_len, batch_size = input.size()
        output= self.embedding(input).view(1, batch_size, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden


# In[43]:


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_SENT_LEN):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        input = input.view(1,-1)
        batch_size = input.size()[1]
        
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0,1))
        
        output = torch.cat((embedded[0], attn_applied.transpose(0,1)[0]), 1)
        
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights


# -----------------------------------------------------------

# ## Train

# In[44]:


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, mode_dec=None, mode_enc=None):
    
    hidden_size = hyper['HIDDEN_SIZE']
    learning_rate = hyper['LR']
    dropout_p = hyper['DROP_OUT']
    teacher_forcing_ratio = hyper['TEACHER_RATIO']
    n_layers = hyper['N_LAYERS']
    ker_size = hyper['KER_SIZE']
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0] 
    target_length = target_tensor.size()[0]
    batch_size = input_tensor.size()[1]
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_outputs = torch.zeros(target_length, batch_size, encoder.hidden_size, device=device) 
    
    loss = 0
    
    # feed-forward layer resulting encoder outputs, ei refers to each word token in input sentence
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    if mode_enc == 'cnn':
        encoder_hidden = nn.Linear(hidden_size,hidden_size)(encoder_hidden[0].cpu()).to(device).unsqueeze(0)
    else:
        encoder_hidden = nn.Linear(2*hidden_size,hidden_size)(
            torch.cat((encoder_hidden[0].cpu(),encoder_hidden[1].cpu()),dim = 1)).to(device).unsqueeze(0)
   # print('encoder_hidden size:', encoder_hidden.size())
#     print("t:", encoder_hidden.dtype)
    decoder_hidden = encoder_hidden.to(device)
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device)  
    #print('input to decoder:', decoder_input.size(), decoder_hidden.size(), encoder_outputs.size())
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for di in range(target_length):
            if mode_dec == 'attn':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  
            loss += criterion(decoder_output, target_tensor[di])    
    else:
  
        for di in range(target_length):
            if mode_dec == 'attn':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.transpose(0,1).detach()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
 
            #decoder_input = topi.squeeze().detach()
    
            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()
    encoder_optimizer.step() 
    decoder_optimizer.step()

    return loss.item() / target_length


# ------------------------------------------------------------

# ## Evaluate

# In[46]:


def convert_idx_2_sent(pred_tensor, truth_tensor,lang_obj):
    pred_word_list = []
    truth_word_list = []
    for i in pred_tensor:
        if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            pred_word_list.append(lang_obj.index2word[i.item()])
    for j in truth_tensor:
        if j.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            truth_word_list.append(lang_obj.index2word[j.item()])
    pred_sent = (' ').join(pred_word_list)
    truth_sent = (' ').join(truth_word_list)
    return pred_sent, truth_sent


# In[47]:


def bleu(corpus, truths):
    '''
    corpus: list, NBs * BATCHSIZE * MAX_LEN
    truths: list, NBs * BATCHSIZE * MAX_LEN
    
    return: array of length NBs, avg blue score for each batch
    '''
    n = len(corpus)
    bleus = [0]*n
    for i in range(n):
        pred, true = corpus[i], truths[i]
        sumbleu = 0.0
        
        for j in range(len(corpus[i])):
            pred_tensor, true_tensor = pred[j], true[j]
            pred_sent, true_sent = convert_idx_2_sent(pred_tensor, true_tensor, target_tra)
            sumbleu += corpus_bleu(true_sent, pred_sent).score
        avgbleu = sumbleu / len(corpus[i])
        bleus[i] = avgbleu
    return bleus


# In[59]:


def bleu_new(corpus,truths):
    n = len(corpus)
    bleu = [0]*n
    for i in range(n):
        pred, true = corpus[i], truths[i]
        pred_ls = [convert_idx_2_sent_new(sent, target_tra) for sent in pred]
        true_ls = [convert_idx_2_sent_new(sent, target_tra) for sent in true]
        bleu[i] = corpus_bleu(pred_ls, [true_ls]).score
    return np.mean(bleu)


# In[49]:


def convert_idx_2_sent_new(idx_tensor, lang_obj):
    word_list = []
    #truth_word_list = []
    for i in idx_tensor:
        if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            word_list.append(lang_obj.index2word[i.item()])
#     for j in truth_tensor:
#         if j.item() not in set([PAD_IDX,EOS_token,SOS_token]):
#             truth_word_list.append(lang_obj.index2word[j.item()])
    sent = (' ').join(word_list)
    #truth_sent = (' ').join(truth_word_list)
    return sent


# In[53]:


def evaluate(encoder, decoder, data_loader, mode_enc, mode_dec, max_length=MAX_SENT_LEN):
    start = time.time()
    hidden_size = hyper['HIDDEN_SIZE']
    learning_rate = hyper['LR']
    dropout_p = hyper['DROP_OUT']
    teacher_forcing_ratio = hyper['TEACHER_RATIO']
    n_layers = hyper['N_LAYERS']
    ker_size = hyper['KER_SIZE']
    encoder.eval()
    decoder.eval()
    inputs = []
    corpus = []
    truths = []
    for i, (input_sentences, target_sentences,len1,len2) in enumerate(data_loader):
#         if i % 5 == 0:
#             print('Time: {}, Step: [{}/{}]'.format(
#                 timeSince(start, i + 1/len(train_loader)), i, len(data_loader)))
        inputs.append(input_sentences.to(device))
        input_tensor = input_sentences.transpose(0,1).to(device)
        truths.append(target_sentences.to(device))
        target_tensor = target_sentences.transpose(0,1).to(device) 
        #truths.append(target_tensor)
        input_length = input_tensor.size()[0]
        batch_size = input_tensor.size()[1]
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        if mode_enc == 'cnn':
            encoder_hidden = nn.Linear(hidden_size,hidden_size)(encoder_hidden[0].cpu()).to(device).unsqueeze(0)
        else:
            encoder_hidden = nn.Linear(2*hidden_size,hidden_size)(
            torch.cat((encoder_hidden[0].cpu(),encoder_hidden[1].cpu()),dim = 1)).to(device).unsqueeze(0)

        decoder_hidden = encoder_hidden.to(device)
        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device) 
        decoded_words = torch.zeros(batch_size, max_length)
        decoder_attentions = torch.zeros(max_length, max_length)
    
        for di in range(max_length):
            if mode_dec == 'attn':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            decoded_words[:,di] = topi.squeeze()
            decoder_input = topi.squeeze().detach()
        corpus.append(decoded_words)
        #print(inputs[0].size(), corpus[0].size(), truths[0].size())
    return inputs, corpus, truths


# ## Training

# In[54]:


def plot_loss_bleu(bleu_score,
                losses):
    
    batches = np.arange(0, len(bleu_score))
    f, axs = plt.subplots(1, 2, figsize=(15,5))
    ax1 = axs[0]
    ax1.plot(batches, losses, label='Validation loss')
    ax1.set_xlabel("number of batches")
#     ax1.plot(batches, validation_loss_history, alpha=0.7, label='Validation Loss')
    ax1.legend(loc='upper right')

    ax2 = axs[1]
    ax2.plot(batches, bleu_score, label='Validation BLEU Score')
    ax2.set_xlabel("number of batches")
#     ax2.plot(batches, validation_acc_history, alpha=0.7, label='Validation Accuracy')
    ax2.legend(loc='upper right')
    plt.show()


# In[58]:


def train_model(mode_enc, mode_dec, hyper, start_epoch=0):
    start = time.time()
    hidden_size = hyper['HIDDEN_SIZE']
    learning_rate = hyper['LR']
    eva_every = hyper['EVA_EVERY']
    dropout_p = hyper['DROP_OUT']
    teacher_forcing_ratio = hyper['TEACHER_RATIO']
    n_layers = hyper['N_LAYERS']
    ker_size = hyper['KER_SIZE']
    num_epoch = hyper['NUM_EPOCHS']
    early_stopping = False
    patience = 3
    required_progress = 0.01
    
    if mode_enc == 'rnn':
        encoder = EncoderRNN(source_tra.n_words, hidden_size).to(device)
        
    else:
        encoder = EncoderCNN(source_tra.n_words, hidden_size, word_vecs=word_vecs[source_tra.name], dropout_p=dropout_p, ker_size=ker_size).to(device)
       
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    
    if mode_dec == 'attn':
            decoder = AttnDecoder(hidden_size, target_tra.n_words, dropout_p=dropout_p).to(device)
    else:
            decoder = Decoder(hidden_size, target_tra.n_words, dropout_p=dropout_p).to(device)
            
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() 
    plot_bleu_score_val = []
    plot_losses = []
    loss_total = 0 
    best_score = None
    count = 0
    filename = '/scratch/rw2268/NMT/models/best_h_' #########
    for epoch in range(1, num_epoch + 1): 
        for i, (input_sentences, target_sentences,len1,len2) in enumerate(train_loader): 
            ### delete break

            encoder.train()
            decoder.train()
            input_tensor = input_sentences.transpose(0,1).to(device)    
            target_tensor = target_sentences.transpose(0,1).to(device)
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, mode_dec=mode_dec, mode_enc=mode_enc)
            loss_total += loss
            if i > 0 and i % eva_every == 0:
                    inputs, corpus, truths = evaluate(encoder, decoder, val_loader, max_length=MAX_SENT_LEN, mode_enc=mode_enc, mode_dec=mode_dec)
                    #bleu_score_val = bleu(corpus, truths)
                    bleu_score_val_avg = bleu_new(corpus, truths)#np.mean(bleu_score_val)
                    loss_avg = loss_total / eva_every
                    loss_total = 0
                    plot_losses.append(loss_avg)
                    plot_bleu_score_val.append(bleu_score_val_avg)
                    if best_score is None:
                        best_score = bleu_score_val_avg
                    if bleu_score_val_avg < best_score + required_progress:
                        count += 1
                    elif bleu_score_val_avg > best_score:
                        state = {'epoch': start_epoch + epoch + 1, 
                                 'state_dict_enc': encoder.state_dict(),
                                 'state_dict_dec': decoder.state_dict(), 
                                 'best_accuracy': best_score}
#                                  'optimizer_enc': encoder_optimizer.state_dict(),
#                                 'optimizer_dec': decoder_optimizer.state_dict()}
                        print ('new best achieved')
                        torch.save(state, filename+str(hidden_size)+'.pth')
                        best_score = bleu_score_val_avg
                        count = 0
                    if early_stopping:
                        if count >= patience:
                            print("earily stop triggered")
                            break
                    print('-----------------------------------------')
                    print('Time: {0}, Epoch: [{1}/{2}], Step: [{3}/{4}], Train Loss: {5}, BLEU score: {6}'.format(
                        timeSince(start, i + 1/len(train_loader)), epoch, num_epoch, i, 
                        len(train_loader), loss_avg, bleu_score_val_avg))
#                     print('\nInput> %s'%(' '.join([source_tra.index2word[i.item()] for i in inputs[0][3] if i.item() not in set([PAD_IDX,EOS_token,SOS_token])])))
#                     print('\nTarget= %s'%(convert_idx_2_sent(truths[0][3],corpus[0][3], target_tra)[0]),
#                     '\nPredict< %s' %(convert_idx_2_sent(truths[0][3],corpus[0][3], target_tra)[1]))
                    print('-----------------------------------------')
        
        if early_stopping:
            if count >= patience:
                break
    plot_loss_bleu(plot_bleu_score_val, plot_losses)
#     torch.save(encoder.state_dict(), 'rnn_encoder_13.pkl')
#     torch.save(decoder.state_dict(), 'attn_rnn_decoder_13.pkl')
    return inputs, corpus, truths


# RNN+Attn

# In[56]:


hs = [256,512,1024]
for h in hs:
    print('hidden=',h)
    hyper['HIDDEN_SIZE'] = h
    train_model('rnn', 'attn', hyper)


# In[46]:


# train_model('rnn', 'attn', hyper)


# # In[64]:


# train_model('rnn', 'attn', hyper)


# # Rnn+noattn

# # In[35]:


# # train_model('rnn', 'noattn', hyper)


# # cnn+attn

# # In[65]:


# train_model('cnn', 'attn', hyper)


# # cnn+noattn

# # In[ ]:


# train_model('cnn', 'noattn', hyper)


# In[ ]:




