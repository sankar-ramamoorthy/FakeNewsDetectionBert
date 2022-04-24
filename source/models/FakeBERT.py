import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, Adadelta

# def initialize_model(device, embed_dim, filter_sizes, num_filters, num_classes,
def initialize_model(device, max_len, filter_sizes, num_filters, num_classes,
                     dropout=0.2, learning_rate=0.01):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = FakeBERTCNN(max_len=max_len,
                            filter_sizes=filter_sizes,
                            num_filters=num_filters,
                            num_classes=num_classes,
                            dropout_p=dropout)
    
    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer

    return cnn_model

class FakeBERTCNN(nn.Module):
    # def __init__(self, pretrained_embedding, emb_dim, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], num_classes=2, dropout_p=0.2):
    # def __init__(self, emb_dim, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], num_classes=2, dropout_p=0.2):
    def __init__(self, max_len=100, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], num_classes=2, dropout_p=0.2):
        super(FakeBERTCNN, self).__init__()

        # self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        # maybe add embedding code here instead of earlier
        # getting embeddings for each batch (vs. full data 1st) could be more efficience for memory

        # CNN
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=max_len,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # MaxPool (this is the same for all pooling)
        self.pool = nn.MaxPool1d(5)

        # ReLU Activation
        self.relu = nn.ReLU()

        # Final Convolutional Layers
        self.conv1d1 = nn.Conv1d(128, 128, 5) # move to forward?
        self.conv1d2 = nn.Conv1d(128, 128, 5)
        self.flat = nn.Flatten()

        # Fully-connected layers and Dropout
        # self.fc1 = nn.Linear(np.sum(num_filters), num_classes) # dimensions wrong
        self.fc1 = nn.Linear(2176, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)

    # def forward(self, input_ids):
    def forward(self, x_embed):
            # # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
            # x_embed = self.embedding(input_ids).float()

            # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
            # Output shape: (b, embed_dim, max_len)
            # print(f'x_embed shape: {x_embed.shape}')
            # x_reshaped = x_embed.permute(0, 2, 1) # doesn't work, x_embed is not 3D
            # x_reshaped = x_embed#x_embed.permute(1,2,0)
            # print(f'x_reshaped shape: {x_reshaped.shape}')

            # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
            # for conv in self.conv1d_list:
            #   print(f'conv weight shape: {conv.weight.shape}')
            x_conv_list = [self.relu(conv1d(x_embed)) for conv1d in self.conv1d_list]
            # for i in range(len(x_conv_list)):
                # print(f'conv {i} dim: {x_conv_list[i].shape}')
            # Max pooling. Output shape: (b, num_filters[i], 1)
            # x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            #     for x_conv in x_conv_list]
            x_pool_list = [self.pool(x_conv) for x_conv in x_conv_list]
            # for i in range(len(x_pool_list)):
                # print(f'pooled {i} dim: {x_pool_list[i].shape}')
            
            # Concatenate x_pool_list to feed the fully connected layer.
            # Output shape: (b, sum(num_filters))
            # x_fc1 = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
            #                 dim=1)
            x_concat = torch.cat([x_pool for x_pool in x_pool_list], dim=2).squeeze(1)
            # print(f'x_concat shape: {x_concat.shape}')
            x_final_conv1 = self.pool(self.relu(self.conv1d1(x_concat)))
            # print(f'x_final_conv1 shape: {self.relu(self.conv1d1(x_concat)).shape}')
            # print(f'x_final_pooled1 shape: {x_final_conv1.shape}')
            x_final_conv2 = self.pool(self.relu(self.conv1d2(x_final_conv1)))
            # print(f'x_final_conv2 shape: {self.relu(self.conv1d2(x_final_conv1)).shape}')
            # print(f'x_final_pooled2 shape: {x_final_conv2.shape}')
            x_flat = self.flat(x_final_conv2)
            # print(f'x_flat shape: {x_flat.shape}')
            
            # Compute logits. Output shape: (b, n_classes)
            x_fc1 = self.fc1(self.dropout(x_flat))
            # print(f'dense 1 shape: {x_fc1.shape}')
            logits = self.fc2(self.dropout(x_fc1))
            # print(f'logits shape: {logits.shape}')
            # logits = nn.Linear(x_fc1.shape[2], self.num_classes)(self.dropout(x_fc1))

            return logits
