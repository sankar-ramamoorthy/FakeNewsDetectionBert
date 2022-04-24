#Ref: https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertModel

def initialize_model(device, dropout = 0.2, hidden = 768, max_len = 64, n_classes = 2):
    """Instantiate a CNN model and an optimizer."""

    # Instantiate model
    model = linearBERT(dropout, hidden, max_len, n_classes)
    
    # Send model to `device` (GPU/CPU)
    model.to(device)

    return model


class linearBERT(nn.Module):

    def __init__(self, dropout=0.5, hidden=768, max_len=64, n_classes=2):

        super(linearBERT, self).__init__()

        #self.embedding = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden * max_len, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x_embed):
        dropout_output = self.dropout(x_embed)
        #print('dropout shape: ', dropout_output.shape)
        flatten_output = self.flatten(dropout_output)
        #print('flatten shape: ',flatten_output.shape)
        linear_output = self.linear(flatten_output)
        #print('linear shape: ', linear_output.shape)
        final_layer = self.sig(linear_output)
        #print('x_fc1 shape: ', final_layer.shape)

        return final_layer