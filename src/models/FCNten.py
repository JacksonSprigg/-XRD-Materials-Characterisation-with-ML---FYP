import torch.nn as nn
import torch.nn.functional as F

# An FCN comparable to CNNten for comparison purposes

class FCNten(nn.Module):
    def __init__(self, num_classes=230):
        super(FCNten, self).__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=12, stride=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv3 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv4 = nn.Conv1d(24, 24, kernel_size=12, stride=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.33)

        self.conv5 = nn.Conv1d(24, 2000, kernel_size=1)
        self.conv6 = nn.Conv1d(2000, num_classes, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))

        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        x = self.gap(x).squeeze(2)
        
        return x