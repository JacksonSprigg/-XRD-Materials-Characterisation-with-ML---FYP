import torch.nn as nn
import torch.nn.functional as F

class CNNten(nn.Module):
    def __init__(self):
        super(CNNten, self).__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=12, stride=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv3 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv4 = nn.Conv1d(24, 24, kernel_size=12, stride=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.33)

        self.fcl1 = nn.Linear(4992,2000)
        self.fcl2 = nn.Linear(2000,230)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))

        x = self.flatten(x)

        x = self.dropout(F.relu(self.fcl1(x)))

        x = self.fcl2(x)
        
        return x