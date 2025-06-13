import torch.nn as nn

class FallGRUClassifier(nn.Module):
    def __init__(self, input_size=104, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])
