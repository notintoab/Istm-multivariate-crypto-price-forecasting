import torch
import torch.nn as nn
from torch.autograd import Variable
from config import DROPOUT

#
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=DROPOUT)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).detach()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).detach()
        
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        out = self.fc(out[:, -1, :])
        
        return out