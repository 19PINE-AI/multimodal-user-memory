import json, glob, sys
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "results/binding_followup/runs"
pat = sys.argv[1]
for f in sorted(glob.glob(str(R / pat))):
    d = json.load(open(f)); m = d["meta"]; lm = d["lm"]
    s = " ".join("M%s:tf=%.3f,any=%.2f,other=%.2f" % (k, v["tf_exact"], v["greedy_any_stored_code"], v["greedy_other_code_exact"]) for k, v in lm.items())
    P = d["probes"].get(str(m["mmax"]), {})
    sl = [round(P["slot%d_linear" % i]["char_acc"], 2) for i in range(m["mmax"]) if "slot%d_linear" % i in P]
    nc = round(P.get("namecond_mlp", {}).get("char_acc", float("nan")), 2)
    I = d.get("interference") or {}
    print(m["tag"], "stop", m.get("final_step"), m.get("stop_reason"), "|", s, "| slot_lin", sl, "namecond_mlp", nc,
          "| R2 %.2f cosABBA %.2f" % (I.get("lin_R2", float("nan")), I.get("cos_AB_BA", float("nan"))))
