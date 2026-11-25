import copy
from pathlib import Path
import torch
from torch.nn.utils import prune
from models import FashionCNN, count_parameters
from engine import train, evaluate, measure_latency, save_checkpoint

def prunable(model):
    return [(m,"weight") for m in model.modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]

def apply_unstructured(model, amount, mode):
    mods=prunable(model)
    if amount:
        if mode=="global": prune.global_unstructured(mods,pruning_method=prune.L1Unstructured,amount=amount)
        else:
            for m,n in mods: prune.l1_unstructured(m,name=n,amount=amount)
    return [(m.weight_orig,m.weight_mask) for m,_ in mods if hasattr(m,"weight_orig")]

def remove_masks(model):
    for m,n in prunable(model):
        if hasattr(m,n+"_orig"): prune.remove(m,n)

def sparsity(model):
    z=t=0
    for m,_ in prunable(model): z+=(m.weight==0).sum().item(); t+=m.weight.numel()
    return z/t

def compact_model(src, fraction):
    device=next(src.parameters()).device
    def keep(w):
        n=w.shape[0]; k=max(1,round(n*(1-fraction)))
        return w.detach().flatten(1).norm(2,1).topk(k).indices.sort().values
    i1,i2,i3=keep(src.conv1.weight),keep(src.conv2.weight),keep(src.conv3.weight)
    dst=FashionCNN((len(i1),len(i2),len(i3)),src.fc1.out_features).to(device)
    with torch.no_grad():
        dst.conv1.weight.copy_(src.conv1.weight[i1]); dst.conv1.bias.copy_(src.conv1.bias[i1])
        dst.conv2.weight.copy_(src.conv2.weight[i2][:,i1]); dst.conv2.bias.copy_(src.conv2.bias[i2])
        dst.conv3.weight.copy_(src.conv3.weight[i3][:,i2]); dst.conv3.bias.copy_(src.conv3.bias[i3])
        ix=torch.cat([torch.arange(int(j)*16,(int(j)+1)*16,device=device) for j in i3])
        dst.fc1.weight.copy_(src.fc1.weight[:,ix]); dst.fc1.bias.copy_(src.fc1.bias)
        dst.fc2.weight.copy_(src.fc2.weight); dst.fc2.bias.copy_(src.fc2.bias)
    return dst

def tensor_bytes(state):
    return sum(v.numel()*v.element_size() for v in state.values())

def sparse_save(model,path):
    state={}
    for k,v in model.state_dict().items():
        state[k]=v.detach().cpu().to_sparse_coo() if v.is_floating_point() and v.ndim>=2 else v.cpu()
    torch.save(state,path); return Path(path).stat().st_size

def run_pruning(base,tr,va,te,cfg,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]
    for mode in ("global","layerwise"):
        for amount in cfg["pruning"]["sparsity_targets"]:
            m=copy.deepcopy(base); masks=apply_unstructured(m,amount,mode)
            if amount: train(m,tr,va,cfg["training"]["fine_tune_epochs"],cfg["training"]["learning_rate"],
                             cfg["training"]["weight_decay"],cfg["device"],masks=masks)
            remove_masks(m); ev=evaluate(m,te,cfg["device"])
            spath=out/f"{mode}_{int(amount*100)}_sparse.pt"
            sb=sparse_save(m,spath); lat=measure_latency(m,te,cfg["device"],cfg["training"]["latency_warmup"],cfg["training"]["latency_iterations"])
            row={"kind":mode,"target_sparsity":amount,"actual_sparsity":sparsity(m),"accuracy":ev["accuracy"],
                 "parameters":count_parameters(m),"dense_state_bytes":tensor_bytes(m.state_dict()),
                 "sparse_file_bytes":sb,"latency_ms_per_image":lat}
            rows.append(row); save_checkpoint(m,out/f"{mode}_{int(amount*100)}.pt")
            print(mode,amount,row["accuracy"],row["actual_sparsity"],lat)
    for amount in cfg["pruning"]["structured_targets"]:
        m=compact_model(base,amount)
        train(m,tr,va,cfg["training"]["fine_tune_epochs"],cfg["training"]["learning_rate"],
              cfg["training"]["weight_decay"],cfg["device"])
        ev=evaluate(m,te,cfg["device"]); lat=measure_latency(m,te,cfg["device"],cfg["training"]["latency_warmup"],cfg["training"]["latency_iterations"])
        row={"kind":"structured","target_sparsity":amount,"actual_sparsity":0.,"accuracy":ev["accuracy"],
             "parameters":count_parameters(m),"dense_state_bytes":tensor_bytes(m.state_dict()),
             "sparse_file_bytes":"","latency_ms_per_image":lat}
        rows.append(row); save_checkpoint(m,out/f"structured_{int(amount*100)}.pt")
    return rows
