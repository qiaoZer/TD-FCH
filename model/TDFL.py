import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TDFL(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TDFL, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # print('TDFL with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        n_batch, c, t, v = x.size()  # Shape: [N, C, T, V]
        assert t == self.n_segment, "The size of time dimension should match the number of segments."
        
        # Adaptive average pooling along the node dimension
        out = F.adaptive_avg_pool2d(x, (t, 1))  # Shape: [N, C, T, 1]
        out = out.view(-1, t)  # Shape: [N*C, T]
        
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)  # Shape: [N*C, 1, T, 1]
        local_activation = self.L(out.view(n_batch, c, t)).view(n_batch, c, t, 1)  # Shape: [N, C, T, 1]
        
        new_x = x * local_activation  # Element-wise multiplication 

        out = F.conv2d(new_x.view(1, n_batch * c, t, v),  # Shape: [1, N*C, T, V]
                       conv_kernel,  # Shape: [N*C, 1, T, 1]
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, v)  # Shape: [N, C, T, V]

        return out + x
