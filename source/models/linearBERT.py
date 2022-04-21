#Ref: https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertModel

def initialize_model(device, dropout=0.2):
    """Instantiate a CNN model and an optimizer."""

    # Instantiate model
    model = linearBERT(dropout)
    
    # Send model to `device` (GPU/CPU)
    model.to(device)

    return model


class linearBERT(nn.Module):

    def __init__(self, dropout=0.5):

        super(linearBERT, self).__init__()

        #self.embedding = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x_embed):
        #x_embed = self.embedding(input_ids)[-1]
        #print('pooled_output shape: ', x_embed.shape)
        dropout_output = self.dropout(x_embed)
        #print('dropout shape: ', dropout_output.shape)
        linear_output = self.linear(dropout_output)
        #print('linear shape: ', linear_output.shape)
        final_layer = self.sig(linear_output)
        #print('x_fc1 shape: ', final_layer.shape)

        return final_layer