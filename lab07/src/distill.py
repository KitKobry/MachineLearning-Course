from pathlib import Path
from models import build_student, count_parameters
from engine import train,evaluate,measure_latency,save_checkpoint
from compression import tensor_bytes

def run_distillation(teacher,tr,va,te,cfg,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]
    for name,use_teacher in [("student_plain",False),("student_distilled",True)]:
        m=build_student(cfg)
        hist=train(m,tr,va,cfg["distillation"]["epochs"],cfg["distillation"]["learning_rate"],
                   cfg["training"]["weight_decay"],cfg["device"],
                   teacher=teacher if use_teacher else None,temperature=cfg["distillation"]["temperature"],
                   alpha=cfg["distillation"]["alpha"])
        ev=evaluate(m,te,cfg["device"]); lat=measure_latency(m,te,cfg["device"],
            cfg["training"]["latency_warmup"],cfg["training"]["latency_iterations"])
        path=out/f"{name}.pt"; save_checkpoint(m,path)
        rows.append({"model":name,"accuracy":ev["accuracy"],"parameters":count_parameters(m),
                     "state_bytes":tensor_bytes(m.state_dict()),"checkpoint_bytes":path.stat().st_size,
                     "latency_ms_per_image":lat,"history":hist})
    return rows
