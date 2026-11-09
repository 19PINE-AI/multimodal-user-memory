s=open("t3_binding.py").read()
old="""                if step >= 8000 and step - best_acc_step >= 4000:"""
new="""                if wl < best * 0.97: best, best_step = wl, step
                # v8: also require the loss to have stalled (<3% gain in 4000 steps); otherwise slow
                # learners (role-keyed, M=3) are cut off while still improving
                if step >= 8000 and step - best_acc_step >= 4000 and step - best_step >= 4000:"""
assert old in s; s=s.replace(old,new)
s=s.replace('stop_rule = "acc" if','stop_rule = "acc2" if')
s=s.replace('if stop_rule == "acc" and anneal_from is None:','if stop_rule == "acc2" and anneal_from is None:')
open("t3_binding.py","w").write(s)
n=open("needs_train.py").read()
old="""if m.get("stop_rule") == "acc":
    sys.exit(1)                      # finished under the accuracy rule (acc_plateau or cap)"""
new="""if m.get("stop_rule") == "acc2":
    sys.exit(1)                      # finished under the acc+loss rule
if m.get("stop_rule") == "acc":
    # v7 rule stopped on accuracy alone: rerun if the loss was still falling >3% in the 4000 steps before the stop
    c = m.get("curve", []); stop = m["final_step"] - 1000
    lb = [r["loss"] for r in c if stop - 4500 <= r["step"] <= stop - 4000]
    le = [r["loss"] for r in c if stop - 500 <= r["step"] <= stop]
    falling = bool(lb and le) and min(le) < min(lb) * 0.97
    print(f"{tag}: acc rule stop@{stop}, loss {min(lb) if lb else 0:.3f}->{min(le) if le else 0:.3f}, falling={falling}")
    sys.exit(0 if falling else 1)"""
assert old in n; n=n.replace(old,new); open("needs_train.py","w").write(n)
print("patched")
