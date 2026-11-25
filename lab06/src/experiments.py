import time,psutil,torch
from torch.profiler import profile,ProfilerActivity,record_function
from torch import nn
from models import build
from engine import evaluate,train
from data import loaders

def profile_epoch(model,tr,cfg,out):
    opt=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"]);ce=nn.CrossEntropyLoss()
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as prof:
        for i,(x,y) in enumerate(tr):
            if i==20: break
            with record_function("data_loading_and_transfer"): x,y=x.to(cfg["device"]),y.to(cfg["device"])
            opt.zero_grad()
            with record_function("forward"): loss=ce(model(x),y)
            with record_function("backward"): loss.backward()
            with record_function("optimizer_step"): opt.step()
    prof.export_chrome_trace(str(out/"torch_profile_trace.json"))
    (out/"torch_profiler_table.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total"),encoding="utf-8")

def run_workers(name,cfg,ds):
    rows=[]
    for w in cfg["studies"]["workers"]:
        tr,_,_=loaders(ds,cfg["data"]["batch_size"],w,cfg["seed"]);t=time.perf_counter();n=0
        for x,y in tr:
            n+=len(y)
            if n>=8192:break
        sec=time.perf_counter()-t;rows.append({"num_workers":w,"seconds":sec,"throughput_samples_s":n/sec})
    return rows

def run_batches(name,cfg,ds):
    rows=[];proc=psutil.Process()
    for bs in cfg["studies"]["batch_sizes"]:
        tr,va,te=loaders(ds,bs,cfg["data"]["num_workers"],cfg["seed"]);m=build(name).to(cfg["device"]);opt=torch.optim.Adam(m.parameters(),lr=cfg["training"]["learning_rate"]);ce=nn.CrossEntropyLoss();peak=proc.memory_info().rss;t=time.perf_counter()
        for _ in range(cfg["studies"]["batch_epochs"]):
            m.train()
            for x,y in tr:
                x,y=x.to(cfg["device"]),y.to(cfg["device"]);opt.zero_grad();ce(m(x),y).backward();opt.step();peak=max(peak,proc.memory_info().rss)
        sec=time.perf_counter()-t;ev=evaluate(m,te,cfg["device"])
        rows.append({"batch_size":bs,"epochs":cfg["studies"]["batch_epochs"],"throughput_samples_s":len(tr.dataset)*cfg["studies"]["batch_epochs"]/sec,"peak_process_memory_mb":peak/1048576,"test_accuracy":ev["accuracy"],"seconds":sec})
    return rows

def run_schedulers(name,cfg,ds):
    rows=[]
    for opt in ("SGD","Adam"):
        for sch in ("StepLR","CosineAnnealingLR","ReduceLROnPlateau"):
            tr,va,te=loaders(ds,cfg["data"]["batch_size"],cfg["data"]["num_workers"],cfg["seed"]);m=build(name)
            h=train(m,tr,va,cfg["studies"]["scheduler_epochs"],cfg,opt,sch)
            ev=evaluate(m,te,cfg["device"]);rows.append({"optimizer":opt,"scheduler":sch,"epochs":cfg["studies"]["scheduler_epochs"],"test_accuracy":ev["accuracy"],"test_loss":ev["loss"],"final_lr":h[-1]["lr"]})
    return rows
