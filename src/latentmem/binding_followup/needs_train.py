"""exit 0 -> (re)train needed; exit 1 -> finished run is acceptable. args: tag cap"""
import sys, torch
from pathlib import Path
tag, cap = sys.argv[1], int(sys.argv[2])
import os
CK = Path(os.environ.get("BINDING_CKPT", Path(__file__).resolve().parents[3] / "runs/binding_followup"))
p = CK / f"{tag}.pt"
if not p.exists(): sys.exit(0)
m = torch.load(p, map_location="cpu")["meta"]
if m.get("lora_rank", 16) != 32 or not m.get("conv"): sys.exit(0)
M = m["mmax"]
if M == 1:
    sys.exit(1 if (m.get("stop_reason") == "plateau" or m.get("steps", 0) >= cap) else 0)
if m.get("stop_rule") == "acc2":
    sys.exit(1)                      # finished under the acc+loss rule
if m.get("stop_rule") == "acc":
    # v7 rule stopped on accuracy alone: rerun if the loss was still falling >3% in the 4000 steps before the stop
    c = m.get("curve", []); stop = m["final_step"] - 1000
    lb = [r["loss"] for r in c if stop - 4500 <= r["step"] <= stop - 4000]
    le = [r["loss"] for r in c if stop - 500 <= r["step"] <= stop]
    falling = bool(lb and le) and min(le) < min(lb) * 0.97
    print(f"{tag}: acc rule stop@{stop}, loss {min(lb) if lb else 0:.3f}->{min(le) if le else 0:.3f}, falling={falling}")
    sys.exit(0 if falling else 1)
if m.get("stop_reason") == "max_steps" and m.get("steps", 0) >= cap:
    sys.exit(1)
# old loss rule fired: was top-M accuracy still rising in the 4000 steps before the plateau?
key = f"tf_M{M}"
pts = [(r["step"], r[key]) for r in m.get("curve", []) if key in r]
plateau = m["final_step"] - 1000
before = [a for s, a in pts if s <= plateau - 4000]
window = [a for s, a in pts if plateau - 4000 < s <= plateau]
rising = bool(window) and (max(window) >= (max(before) if before else 0.0) + 0.01)
print(f"{tag}: old loss rule, plateau@{plateau}, best_before={max(before) if before else 0:.3f}, best_window={max(window) if window else 0:.3f}, rising={rising}")
sys.exit(0 if rising else 1)
