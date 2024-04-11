import torch
import torch.nn as nn

# Define Residual Block (for Generator)
class ResidualBlock(nn.Module):
    '''Residual Block: 2 Conv_layer, fixed dimension'''
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self,x):
        return x + self.conv_block(x)


# Define the Generator
class Generator(nn.Module):
    def __init__(self,scale_factor=2,n_residual_blocks=16):
        super(Generator,self).__init__()

        # First Layer: 1 Conv_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU()
        )

        # 16 Residual Blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second Conv Layer Post Residual Blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64, 0.8)
        )

        # 2 Upsampling Layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                # nn.BatchNorm2d(256),
                nn.PixelShuffle(scale_factor),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final Output Layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4), 
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

# Define Downsampling Block (for Discriminator)
class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel, stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
    
# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # First Layer: 1 Conv_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )

        # 7 Downsampling Layers
        self.down = nn.Sequential(
            # 1.
            DownSample(64, 64, stride=2, padding=1),
            # 2.
            DownSample(64, 128, stride=1, padding=1),
            # 3.
            DownSample(128, 128, stride=2, padding=1),
            # 4.
            DownSample(128, 256, stride=1, padding=1),
            # 5.
            DownSample(256, 256, stride=2, padding=1),
            # 6.
            DownSample(256, 512, stride=1, padding=1),
            # 7.
            DownSample(512, 512, stride=2, padding=1)
        )

        # Dense Layer: 1 AvgPool + 2 Conv_layers
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512,1024,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(1024,1,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        out = self.dense(x)
        return out


if __name__ == 'main':
    '''Test the Generator and Discriminator before training'''
    # g = Generator()
    # x = torch.rand([1,3,64,64])
    # print(g(x).shape)
    d = Discriminator()
    x = torch.rand([2,3,512,512])
    print(d(x).shape)