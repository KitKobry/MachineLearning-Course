import csv, json, sys
from pathlib import Path
import yaml, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LAB=HERE.parent
sys.path.insert(0,str(HERE))
from models import build_baseline,count_parameters
from engine import set_seed,make_loaders,train,evaluate,measure_latency,save_checkpoint
from compression import run_pruning,tensor_bytes
from distill import run_distillation

def dump(obj,path):
    path.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def main():
    cfg=yaml.safe_load((LAB/"config.yaml").read_text(encoding="utf-8"))
    cfg["device"]="cpu" if not torch.cuda.is_available() else cfg.get("device","cuda")
    set_seed(cfg["seed"])
    results=LAB/"results"; figs=LAB/"figures"
    results.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    tr,va,te=make_loaders(cfg)
    # 1. Oversized baseline
    teacher=build_baseline(cfg)
    print("Baseline parameter count:",f"{count_parameters(teacher):,}")
    hist=train(teacher,tr,va,cfg["training"]["epochs"],cfg["training"]["learning_rate"],
               cfg["training"]["weight_decay"],cfg["device"])
    ev=evaluate(teacher,te,cfg["device"])
    lat=measure_latency(teacher,te,cfg["device"],cfg["training"]["latency_warmup"],
                        cfg["training"]["latency_iterations"])
    ckpt=results/"baseline.pt"; save_checkpoint(teacher,ckpt)
    baseline={"accuracy":ev["accuracy"],"loss":ev["loss"],"parameters":count_parameters(teacher),
              "state_bytes":tensor_bytes(teacher.state_dict()),"checkpoint_bytes":ckpt.stat().st_size,
              "latency_ms_per_image":lat,"history":hist}
    dump(baseline,results/"baseline.json")
    # 2. Global/layer-wise and structured pruning
    pruned=run_pruning(teacher,tr,va,te,cfg,results/"pruned")
    dump(pruned,results/"pruning_results.json")
    with (results/"pruning_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(pruned[0])); w.writeheader(); w.writerows(pruned)
    plt.figure(figsize=(8,5))
    for kind in ("global","layerwise","structured"):
        rows=sorted([r for r in pruned if r["kind"]==kind],key=lambda x:x["target_sparsity"])
        if rows: plt.plot([r["target_sparsity"]*100 for r in rows],[r["accuracy"] for r in rows],
                          marker="o",label=kind)
    plt.xlabel("Target pruning fraction (%)"); plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST: accuracy vs pruning level"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(figs/"accuracy_vs_sparsity.png",dpi=160); plt.close()
    # 3. Distillation against same student trained without teacher
    students=run_distillation(teacher,tr,va,te,cfg,results/"students")
    dump(students,results/"distillation_results.json")
    with (results/"distillation_results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=[k for k in students[0] if k!="history"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in students: w.writerow({k:v for k,v in row.items() if k!="history"})
    # 4. Data-backed report; energy is deliberately not fabricated.
    best=max(students,key=lambda x:x["accuracy"])
    lines=["# Lab 7 — Compression: pruning and distillation","",
      "## Experimental setup","",
      "- Dataset: Fashion-MNIST; official test split; 10% of training data reserved for validation.",
      f"- Device: {cfg['device']}; random seed: {cfg['seed']}; batch size: {cfg['data']['batch_size']}.",
      "- All accuracies below are measured on the official test split.","",
      "## 1. Oversized baseline CNN","",
      f"- Test accuracy: **{baseline['accuracy']:.4f}**",
      f"- Parameters: **{baseline['parameters']:,}**",
      f"- State tensor bytes: **{baseline['state_bytes']:,}**",
      f"- Checkpoint bytes: **{baseline['checkpoint_bytes']:,}**",
      f"- CPU latency: **{baseline['latency_ms_per_image']:.4f} ms/image**","",
      "## 2. Magnitude and structured pruning","",
      "| Method | Target | Actual unstructured sparsity | Test accuracy | Parameters | Dense bytes | Sparse file bytes | ms/image |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pruned:
        sparse="—" if r["sparse_file_bytes"]=="" else f"{r['sparse_file_bytes']:,}"
        lines.append(f"| {r['kind']} | {r['target_sparsity']:.0%} | {r['actual_sparsity']:.1%} | {r['accuracy']:.4f} | {r['parameters']:,} | {r['dense_state_bytes']:,} | {sparse} | {r['latency_ms_per_image']:.4f} |")
    lines += ["","Global pruning ranks all eligible weights by absolute magnitude. Layer-wise pruning applies the target independently to each prunable layer. Both use fixed masks during three fine-tuning epochs. Structured pruning ranks output filters by L2 norm and rebuilds smaller convolution layers, propagating retained channel indices into the next layer and first fully connected layer.","",
      "### Storage and latency analysis","",
      "Unstructured zero weights do not automatically accelerate ordinary dense CPU convolution or matrix multiplication: dense kernels still traverse dense tensor dimensions unless sparse-aware kernels and supported sparsity patterns are used. COO serialization stores indices as well as values, so it can be larger than dense storage at low sparsity. The sparse checkpoint is a storage comparison, not the normal dense deployment checkpoint. Structured channel removal changes tensor dimensions and is more likely to reduce dense-kernel work.","",
      "## 3. Knowledge distillation","",
      "| Student | Test accuracy | Parameters | State bytes | Checkpoint bytes | CPU ms/image |","|---|---:|---:|---:|---:|---:|"]
    for r in students:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['parameters']:,} | {r['state_bytes']:,} | {r['checkpoint_bytes']:,} | {r['latency_ms_per_image']:.4f} |")
    lines += ["",f"Distillation uses T={cfg['distillation']['temperature']} and alpha={cfg['distillation']['alpha']}. Its objective is alpha × CE(student, labels) + (1 − alpha) × T² × KL(softmax(student/T) || softmax(teacher/T)). The teacher is the trained oversized baseline. The plain student uses the same compact architecture but trains only with hard-label cross-entropy.","",
      "## 4. Efficiency pillars (Chapter 9)","",
      "| Pillar | Evidence | Trade-off / interpretation |","|---|---|---|",
      "| Data | Fashion-MNIST, consistent preprocessing and fixed train/validation/test split | Same data conditions support fair comparisons; data/input size constrain the task. |",
      "| Model | Baseline, unstructured masks, physically compact structured models, plain and distilled students | Compression reduces parameters/storage but may reduce capacity; distillation transfers teacher information. |",
      "| Compute | CPU latency, parameter counts, dense and sparse checkpoint sizes | Unstructured sparsity may not speed dense kernels; smaller structured models can reduce actual operations. |",
      "| Energy | No direct CPU energy sensor is exposed in this environment | Energy is not claimed as measured. A latency × assumed power value would only be a rough proxy, not joules measured at the processor. |",
      "","## 5. Conclusion","",
      f"The better student in this run was **{best['model']}** (test accuracy **{best['accuracy']:.4f}**, {best['parameters']:,} parameters). The preferred compression method should be selected from the measured accuracy, storage and latency trade-offs in the tables and curve; results are hardware- and runtime-dependent.","",
      "## Reproducibility notes","",
      "- Latency is per image, measured by repeated forward passes on batches of up to 64 after warm-up.",
      "- Energy is not directly measured; no fabricated energy values are reported.",
      "- All numeric results are generated by `src/main.py` and stored in `results/`."]
    (LAB/"report.md").write_text("\\n".join(lines),encoding="utf-8")
    print("\\nLab 7 complete. See lab07/report.md, results/, and figures/.")

if __name__=="__main__": main()
