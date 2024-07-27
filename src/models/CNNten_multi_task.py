import torch.nn as nn
import torch.nn.functional as F

class CNNten_multi_task(nn.Module):
    def __init__(self):
        super(CNNten_multi_task, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv1d(1, 24, kernel_size=12, stride=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv3 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv4 = nn.Conv1d(24, 24, kernel_size=12, stride=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.33)

        self.fcl1 = nn.Linear(4992, 2000)

        # Task-specific layers
        self.fcl_spg = nn.Linear(2000, 230)         # Space group classification
        self.fcl_crysystem = nn.Linear(2000, 7)     # Crystal system classification
        self.fcl_blt = nn.Linear(2000, 6)           # Bravais lattice type classification
        self.fcl_composition = nn.Linear(2000, 118)  # Composition prediction

        self.flatten = nn.Flatten()

    def forward(self, x):
        # Shared layers
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))

        x = self.flatten(x)

        x = self.dropout(F.relu(self.fcl1(x)))

        # Task-specific outputs
        out_spg = self.fcl_spg(x)
        out_crysystem = self.fcl_crysystem(x)
        out_blt = self.fcl_blt(x)
        out_composition = self.fcl_composition(x)
        
        return {
            'spg': out_spg,
            'crysystem': out_crysystem,
            'blt': out_blt,
            'composition': out_composition
        }