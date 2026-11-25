import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)
class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)); self.fc=nn.Linear(32*7*7,10)
    def forward(self,x): return self.fc(self.features(x).flatten(1))
class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Linear(49,64); layer=nn.TransformerEncoderLayer(64,4,128,batch_first=True); self.enc=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(64,10)
    def forward(self,x):
        b=x.size(0); x=x.unfold(2,7,7).unfold(3,7,7).contiguous().permute(0,2,3,1,4,5).reshape(b,16,49)
        return self.fc(self.enc(self.proj(x)).mean(1))
def build(name):
    return {"mlp":MLP,"cnn":CNN,"attention":Attention}[name]()
def nparams(m): return sum(p.numel() for p in m.parameters())
