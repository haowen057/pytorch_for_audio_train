from torch import nn
from torchsummary import summary
import torch.optim as optim

# Define Network ,succed from nn.Module
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4CNN add Batch Normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )

        # Defin output & Flatten 
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # Enhance & Linear
        self.linear = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, 10) 

    # forward 
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)  # Use self.global_pool limit output size
        x = self.flatten(x)
        x = self.linear(x)    # Liner
        logits = self.fc2(x)
        return logits


# main program
if __name__ == "__main__":
    cnn = CNNNetwork().cuda()
    # Print model summary information
    summary(cnn, (1, 64, 44))
    
    # Use AdamW as optimal，and add learn rate & weight_decay
    optimizer = optim.AdamW(
        cnn.parameters(),   
        lr=0.001,
        weight_decay=1e-4
        )
