from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def datasets_from_config(cfg):
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,),(0.3530,))])
    root=Path(__file__).resolve().parents[2]/"data"/"FashionMNIST"
    full=datasets.FashionMNIST(str(root),train=True,download=True,transform=tf)
    n=int(cfg["data"]["validation_size"])
    tr,va=random_split(full,[len(full)-n,n],generator=torch.Generator().manual_seed(cfg["seed"]))
    te=datasets.FashionMNIST(str(root),train=False,download=True,transform=tf)
    return tr,va,te

def loaders(ds,batch_size,workers,seed=42):
    tr,va,te=ds
    kw=dict(batch_size=batch_size,num_workers=workers,persistent_workers=workers>0)
    return (DataLoader(tr,shuffle=True,generator=torch.Generator().manual_seed(seed),**kw),
            DataLoader(va,shuffle=False,**kw),DataLoader(te,shuffle=False,**kw))
