import torch.nn as nn
from object_boxes import Detect

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Block, self).__init__()
        
        """
        Block
            Convolutional Layer 
            SiLu Activation
            Depthwise Convolution 
            Batch Normalization
            ReLU Activation
            Depthwise Transpose Convolution
        """
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.silu = nn.SiLU()
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=out_channels)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.depthwise_transpose_conv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding, groups=out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.silu(out)
        out = self.depthwise_conv(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.depthwise_transpose_conv(out)
        return out + x
    
class Aam(nn.Module):

    """
    Convolutional Layer
    Batch Normalization
    ReLU Activation or SiLu Activation
    Block
    Convolutional Layer  - Classification Head
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_class, input_size):
        super(Aam, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

        self.block = Block(in_channels, out_channels, kernel_size, stride, padding)

        self.bbox = Detect()
        self.fc_cl = nn.Linear(out_channels * input_size * input_size, num_class)

    def forward(self, x):

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.block(x)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.block(x)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.block(x)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = x.view(-1, 512*256*256)
        cls = self.fc_cl(x)

        bbox = self.bbox(x)

        return cls, bbox