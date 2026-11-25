import csv,json,sys,yaml,torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;sys.path.insert(0,str(HERE))
from data import datasets_from_config,loaders
from models import build,nparams
from engine import seed,train,evaluate
from experiments import profile_epoch,run_workers,run_batches,run_schedulers

def csvout(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsout(path,obj):path.write_text(json.dumps(obj,indent=2),encoding="utf-8")
def main():
    c=yaml.safe_load((LAB/"config.yaml").read_text());c["device"]="cuda" if c["device"]=="cuda" and torch.cuda.is_available() else "cpu";seed(c["seed"])
    out=LAB/"results";fig=LAB/"figures";out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);ds=datasets_from_config(c)
    # Select the best of the three exact Lab 3 candidate architectures on validation accuracy.
    selection={}
    for name in ("mlp","cnn","attention"):
        tr,va,te=loaders(ds,c["data"]["batch_size"],c["data"]["num_workers"],c["seed"]);m=build(name)
        h=train(m,tr,va,c["training"]["epochs"],c,ckpt=out/f"selection_{name}.pt")
        selection[name]={"parameters":nparams(m),"best_val_accuracy":max(x["val_accuracy"] for x in h),"history":h}
    best=max(selection,key=lambda k:selection[k]["best_val_accuracy"]);c["model"]["selected"]=best;jsout(out/"lab3_model_selection.json",{"selected":best,"candidates":selection});print("Selected Lab 3 model:",best)
    tr,va,te=loaders(ds,c["data"]["batch_size"],c["data"]["num_workers"],c["seed"]);m=build(best)
    hist=train(m,tr,va,c["training"]["epochs"],c,ckpt=out/"best_model_checkpoint.pt");ev=evaluate(m,te,c["device"])
    jsout(out/"best_model_metrics.json",{"architecture":best,"parameters":nparams(m),"test":ev,"history":hist})
    # Optional MLflow logging; core run remains executable if mlflow is not installed.
    try:
        import mlflow
        mlflow.set_tracking_uri(c["mlflow"]["tracking_uri"]);mlflow.set_experiment(c["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="best_lab3_model"):
            mlflow.log_params({"architecture":best,"seed":c["seed"],"batch_size":c["data"]["batch_size"],"epochs":c["training"]["epochs"],"parameters":nparams(m)})
            for row in hist:mlflow.log_metrics({k:v for k,v in row.items() if k!="epoch"},step=row["epoch"])
            mlflow.log_metrics({"test_accuracy":ev["accuracy"],"test_loss":ev["loss"]});mlflow.log_artifact(str(out/"best_model_checkpoint.pt"))
    except Exception as e:print("MLflow logging skipped/unavailable:",e)
    profile_epoch(m,tr,c,out)
    workers=run_workers(best,c,ds);csvout(out/"worker_study.csv",workers)
    plt.figure();plt.plot([r["num_workers"] for r in workers],[r["throughput_samples_s"] for r in workers],marker="o");plt.xlabel("num_workers");plt.ylabel("samples/s");plt.title("DataLoader worker throughput");plt.grid(alpha=.3);plt.savefig(fig/"worker_throughput.png",dpi=160);plt.close()
    batches=run_batches(best,c,ds);csvout(out/"batch_size_study.csv",batches)
    sched=run_schedulers(best,c,ds);csvout(out/"optimizer_scheduler_study.csv",sched)
    # Effective batch 256 via micro-batch 64 x accumulation 4, compared with physical batch 256.
    tr64,va64,_=loaders(ds,64,c["data"]["num_workers"],c["seed"]);a=build(best);ha=train(a,tr64,va64,5,c,accum=4)
    tr256,va256,_=loaders(ds,256,c["data"]["num_workers"],c["seed"]);b=build(best);hb=train(b,tr256,va256,5,c)
    jsout(out/"gradient_accumulation.json",{"effective_batch":256,"micro_batch":64,"accumulation_steps":4,"accumulated":ha,"physical_batch":hb})
    # Explicit interruption at epoch 3, then restore optimizer/model state and continue to epoch 10.
    ck=out/"resume_checkpoint.pt";tr,va,_=loaders(ds,c["data"]["batch_size"],c["data"]["num_workers"],c["seed"]);x=build(best);h3=train(x,tr,va,3,c,ckpt=ck)
    resumed=build(best);hr=train(resumed,tr,va,c["studies"]["resume_total_epochs"],c,ckpt=ck,resume=True)
    jsout(out/"resume_training.json",{"interrupted_after":3,"resumed_to":c["studies"]["resume_total_epochs"],"history":hr})
    # Proposal uses supplied student identity.
    p=c["proposal"]
    proposal=f"""# Individual Project Proposal — Lightweight Fashion-MNIST Classifier

**Student:** {p['student_name']}  
**Student ID:** {p['student_id']}

## Problem
Build a compact image classifier for ten clothing categories that can run on a low-resource CPU device.

## Dataset
Fashion-MNIST: 60,000 official training images and 10,000 test images, 10 classes, 28×28 grayscale. Reserve 5,000 training examples for validation. Normalize using mean 0.2860 and standard deviation 0.3530.

## Success metric
Primary: test accuracy target ≥90%. Secondary: per-image CPU latency, model file size and peak memory.

## Deployment constraints
Latency ≤50 ms per image; serialized model size ≤5 MB; CPU inference without GPU.

## Planned optimization
Start from the best Lab 3 architecture selected by validation accuracy. Evaluate structured pruning and dynamic quantization where supported. Compare accuracy, latency and actual serialized size before/after optimization; retain an uncompressed baseline.

## Risks
CPU timing varies with system load. Use warm-up and repeated measurements. Quantization support depends on layer types, so validate the exported artifact instead of assuming theoretical savings.
"""
    (LAB/"individual_project_proposal.md").write_text(proposal,encoding="utf-8")
    # Summary report assembled from measured artifacts.
    lines=["# Lab 6 — Training pipelines and project proposal","",f"Selected Lab 3 model: **{best}**; parameters: **{nparams(m):,}**; test accuracy: **{ev['accuracy']:.4f}**.","",
    "## Profiling","See `results/torch_profiler_table.txt` and `results/torch_profile_trace.json` for PyTorch profiler output.","",
    "## DataLoader workers","| workers | seconds | samples/s |","|---:|---:|---:|"]
    lines += [f"| {r['num_workers']} | {r['seconds']:.3f} | {r['throughput_samples_s']:.1f} |" for r in workers]
    lines += ["","## Batch-size study","| batch | epochs | samples/s | peak process MB | test accuracy |","|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['batch_size']} | {r['epochs']} | {r['throughput_samples_s']:.1f} | {r['peak_process_memory_mb']:.1f} | {r['test_accuracy']:.4f} |" for r in batches]
    lines += ["","## Optimizer and scheduler","| optimizer | scheduler | epochs | accuracy | loss | final LR |","|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['optimizer']} | {r['scheduler']} | {r['epochs']} | {r['test_accuracy']:.4f} | {r['test_loss']:.4f} | {r['final_lr']:.2e} |" for r in sched]
    lines += ["","## Gradient accumulation","Batch 64 accumulated over 4 steps emulates effective batch 256. Compare both histories in `results/gradient_accumulation.json`.","",
    "## Resume","Training was stopped after epoch 3 and resumed from a checkpoint containing model and optimizer state. See `results/resume_training.json`.","",
    "## Individual project proposal","See `individual_project_proposal.md`.","",
    "## Conclusions","1. YAML configuration and fixed seeds improve reproducibility.","2. Worker count and batch size require measurement; more is not automatically faster on CPU.","3. Faithful resume requires optimizer state and epoch history, not weights alone."]
    (LAB/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Lab 6 completed. Review report.md, results/, figures/, and proposal.")
if __name__=="__main__":main()
