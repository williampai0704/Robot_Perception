import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO ...
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(384,128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(192,64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(96,32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(48,16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.interpolate1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv1x1 = nn.Conv2d(16, 6, kernel_size=1, padding=0)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO ...
        x1 = F.relu(self.conv1(x))  
        x1_down = self.pool1(x1)
        x2 = F.relu(self.conv2(x1_down))  
        x2_down = self.pool1(x2)
        x3 = F.relu(self.conv3(x2_down))  
        x3_down = self.pool1(x3)
        x4 = F.relu(self.conv4(x3_down))  
        x4_down = self.pool1(x4)
        x5 = F.relu(self.conv5(x4_down)) 
        x5_up = self.interpolate1(x5)
        x6 = F.relu(self.conv6(torch.cat((x4,x5_up),dim = 1)))
        x6_up = torch.cat((self.interpolate1(x6), x3),dim = 1)
        x7 = F.relu(self.conv7(x6_up))
        x7_up = torch.cat((self.interpolate1(x7),x2),dim = 1)
        x8 = F.relu(self.conv8(x7_up))
        x8_up = torch.cat((self.interpolate1(x8) , x1),dim = 1)
        x9 = F.relu(self.conv9(x8_up))
        output = self.conv1x1(x9)

        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
