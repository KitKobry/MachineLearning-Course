import random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_loaders(cfg):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(parents=True, exist_ok=True)
    ds = datasets.FashionMNIST(str(root), train=True, download=True, transform=tf)
    nv = int(len(ds) * cfg["data"]["validation_fraction"])
    tr, va = random_split(ds, [len(ds)-nv, nv], generator=torch.Generator().manual_seed(cfg["seed"]))
    te = datasets.FashionMNIST(str(root), train=False, download=True, transform=tf)
    kw = dict(batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"])
    return DataLoader(tr, shuffle=True, **kw), DataLoader(va, shuffle=False, **kw), DataLoader(te, shuffle=False, **kw)

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval(); correct = total = loss_sum = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x,y in loader:
        x,y=x.to(device),y.to(device); out=model(x)
        loss_sum += criterion(out,y).item(); correct += (out.argmax(1)==y).sum().item(); total += y.numel()
    return {"accuracy": correct/total, "loss": loss_sum/total}

def train(model, train_loader, val_loader, epochs, lr, wd, device="cpu",
          masks=None, teacher=None, temperature=4., alpha=.5):
    model.to(device)
    if teacher is not None: teacher.to(device).eval()
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd); ce=nn.CrossEntropyLoss(); hist=[]
    for ep in range(epochs):
        model.train(); total_loss=seen=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); logits=model(x)
            if teacher is None: loss=ce(logits,y)
            else:
                with torch.no_grad(): tlog=teacher(x)
                hard=ce(logits,y)
                soft=nn.functional.kl_div(nn.functional.log_softmax(tlog/temperature,dim=1),
                    nn.functional.softmax(logits/temperature,dim=1),reduction="batchmean")*temperature**2
                loss=alpha*hard+(1-alpha)*soft
            loss.backward(); opt.step()
            if masks:
                with torch.no_grad():
                    for p,m in masks: p.mul_(m)
            total_loss += loss.item()*len(y); seen += len(y)
        m=evaluate(model,val_loader,device)
        hist.append({"epoch":ep+1,"train_loss":total_loss/seen,**m})
        print(f"epoch {ep+1}/{epochs}: loss={total_loss/seen:.4f}, val_acc={m['accuracy']:.4f}")
    return hist

def measure_latency(model, loader, device="cpu", warmup=20, iterations=100):
    model.eval().to(device); x,_=next(iter(loader)); x=x[:min(64,len(x))].to(device)
    with torch.no_grad():
        for _ in range(warmup): model(x)
        start=time.perf_counter()
        for _ in range(iterations): model(x)
    return (time.perf_counter()-start)/iterations/len(x)*1000

def save_checkpoint(model,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),path)
