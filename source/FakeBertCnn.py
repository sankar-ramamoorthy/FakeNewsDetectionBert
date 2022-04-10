import torch
from torch import nn

class FakeBertCnn(nn.Module):
    def __init__(self,
            i_device,
            i_embedding=1000,
            conv_count=3,
            i_conv_input_size=[[1000, 100], [199, 128], [39, 128] ],
            i_conv_output_size=[[996, 128], [195, 128], [35, 128] ],
            i_kernel_size=[5, 5, 5],
            i_stride=[1, 1, 1],
            i_max_pool_size=[5, 5, 5],
            i_flat_size=128,
            i_dense_output=2):
        self.embedding = i_embedding
        self.conv_input_sizes = i_conv_input_size
        self.conv_output_sizes = i_conv_output_size
        self.kernels = i_kernel_size
        self.cnn_layers = []
        for i in range(conv_count):
            self.cnn_layers.extend([
                nn.Conv1d(i_conv_input_size[i][-1],
                          i_conv_output_size[i][-1],
                          i_kernel_size[i],
                          stride=i_stride[i],
                          device=i_device),
                nn.MaxPool1d(i_max_pool_size[i])
                ])

    def forward(self, x):
        next_input = x
        loss = None
        for layer in self.cnn_layers:
            next_input = layer.forward(next_input)

        return next_input

if __name__ == '__main__':
    FakeBertCnn(torch.device('cpu'))
