import random,time
import numpy as np, torch
from torch import nn
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def evaluate(m,loader,device):
    m.eval();n=ok=loss=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device);z=m(x);loss+=nn.functional.cross_entropy(z,y,reduction="sum").item();ok+=(z.argmax(1)==y).sum().item();n+=len(y)
    return {"loss":loss/n,"accuracy":ok/n}
def optimizer(name,m,lr,wd):
    return torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd) if name=="Adam" else torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=wd)
def save(path,m,opt,sch,epoch,hist):
    torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict() if sch else None,"epoch":epoch,"history":hist},path)
def train(m,tr,va,epochs,cfg,optname="Adam",schedname=None,accum=1,ckpt=None,resume=False):
    device=cfg["device"];m.to(device);opt=optimizer(optname,m,cfg["training"]["learning_rate"],cfg["training"]["weight_decay"])
    sch={"StepLR":torch.optim.lr_scheduler.StepLR(opt,3,.5),"CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs),"ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2,factor=.5)}.get(schedname)
    hist=[];start=0
    if resume and ckpt:
        c=torch.load(ckpt,map_location=device,weights_only=False);m.load_state_dict(c["model"]);opt.load_state_dict(c["optimizer"]); 
        if sch and c["scheduler"]: sch.load_state_dict(c["scheduler"])
        start=c["epoch"];hist=c["history"]
    ce=nn.CrossEntropyLoss()
    for ep in range(start,epochs):
        m.train();tot=seen=0;opt.zero_grad(set_to_none=True)
        for i,(x,y) in enumerate(tr):
            x,y=x.to(device),y.to(device);loss=ce(m(x),y)/accum;loss.backward()
            if (i+1)%accum==0 or i+1==len(tr): opt.step();opt.zero_grad(set_to_none=True)
            tot+=loss.item()*accum*len(y);seen+=len(y)
        v=evaluate(m,va,device);row={"epoch":ep+1,"train_loss":tot/seen,"val_loss":v["loss"],"val_accuracy":v["accuracy"],"lr":opt.param_groups[0]["lr"]};hist.append(row)
        if sch: sch.step(v["loss"]) if schedname=="ReduceLROnPlateau" else sch.step()
        if ckpt: save(ckpt,m,opt,sch,ep+1,hist)
        print(f"epoch {ep+1}/{epochs}: loss={row['train_loss']:.4f} val_acc={v['accuracy']:.4f}")
    return hist
