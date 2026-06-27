"""Block until the GPU has enough free memory, then return.

The pilot trains a teacher+student forward over a frozen 3B model; we need
roughly 30 GB free. The box is currently shared, so train.py calls this first
instead of OOM-ing on launch.
"""
import subprocess
import sys
import time


def free_mib() -> int:
    """Free VRAM on GPU 0, in MiB (max over visible GPUs)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return max(int(x) for x in out)


def wait_for_gpu(need_gb: float = 30.0, poll_s: int = 30, timeout_s: int = 0) -> None:
    need = int(need_gb * 1024)
    t0 = time.time()
    while True:
        free = free_mib()
        if free >= need:
            print(f"[gpu_wait] {free} MiB free >= {need} MiB needed — proceeding.", flush=True)
            return
        waited = int(time.time() - t0)
        print(f"[gpu_wait] {free} MiB free < {need} MiB needed; waited {waited}s, polling…",
              flush=True)
        if timeout_s and waited > timeout_s:
            print(f"[gpu_wait] timeout after {waited}s — aborting.", flush=True)
            sys.exit(2)
        time.sleep(poll_s)


if __name__ == "__main__":
    need = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    wait_for_gpu(need)
