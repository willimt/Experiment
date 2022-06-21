import torch
import math
from d2l import torch as d2l
from torch import nn
from MyKu import processing
import nltk
import pandas as pd

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True, dropout=0.5)
        self.decoder = nn.Linear(num_hiddens * 4, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


processing.create_exist2021()
batch_size = 128

# train_iter, test_iter, vocab = processing.load_exist2021_data(batch_size)
data = pd.read_csv(processing.TRAIN_EXIST2021, sep='\t')
print(data)