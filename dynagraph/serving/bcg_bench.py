"""SGLang diffusion (fake weights): eager vs breakable CUDA graph, per request, several resolutions and prompt lengths.

  BCG=1 WARM_RES=512x512,1024x1024 python bcg_bench.py /fake/Z-Image-Turbo
Prints per request: resolution, prompt tokens (approx words), denoise step median (ms), end-to-end (s).
"""
import json
import os
import statistics
import sys
import time

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

MODEL = sys.argv[1]
BCG = os.environ.get("BCG") == "1"
STEPS = int(os.environ.get("STEPS", "9"))
WARM_RES = os.environ.get("WARM_RES", "512x512,1024x1024").split(",")
SHORT = "a red cube on a white table"
LONG = " ".join(["A detailed cinematic scene of a glass observatory above a quiet lake at sunrise, with soft mist,"
                 " warm reflections, crisp architectural detail, and birds in the distance."] * 6)
A, B, MISS = os.environ.get("RES", "512x512,1024x1024,768x1344").split(",")
GUIDANCE = float(os.environ.get("GUIDANCE", "0.0"))
REQS = [
    (A, SHORT), (A, SHORT), (A, LONG), (A, LONG),
    (B, SHORT), (B, LONG),
    (MISS, SHORT),  # not in the warmup list
    (A, SHORT),
]
if os.environ.get("VARPROMPT") == "1":
    # every request a different prompt length, as in real traffic
    import random as _r

    _rng = _r.Random(0)
    _words = (LONG + " " + LONG).split()
    _n = int(os.environ.get("NREQ", "16"))
    _lens = _rng.sample(range(4, 160), _n)
    REQS = [(A if i < _n - 4 or _n > 16 else B, " ".join(_words[:n])) for i, n in enumerate(_lens)]

def run(gen, res, prompt):
    w, h = (int(x) for x in res.split("x"))
    t = time.perf_counter()
    r = gen.generate(sampling_params_kwargs=dict(prompt=prompt, width=w, height=h, seed=0, num_inference_steps=STEPS,
                                                 guidance_scale=GUIDANCE, save_output=False))
    dt = time.perf_counter() - t
    r = r[0] if isinstance(r, list) else r
    m = r.metrics or {}
    steps = list(m.get("steps") or [])
    return dt, steps, m


def main():
    kw = dict(model_path=MODEL, num_gpus=1, master_port=int(os.environ.get("MASTER_PORT", "30005")), enable_torch_compile=os.environ.get("COMPILE") == "1", dit_layerwise_offload=False, dit_cpu_offload=False)
    if BCG:
        kw.update(enable_breakable_cuda_graph=True, warmup_resolutions=WARM_RES)
    t0 = time.perf_counter()
    gen = DiffGenerator.from_pretrained(**kw)
    print(f"[init] {time.perf_counter() - t0:.1f} s (BCG={BCG})", flush=True)



    # un-timed warmup at the first resolution (JIT, autotune, first-touch)
    for _ in range(2):
        run(gen, A, SHORT)
    out = []
    for res, p in REQS:
        dt, steps, m = run(gen, res, p)
        med = statistics.median(steps) if steps else float("nan")
        print(f"[req] {res:>9} prompt={len(p.split()):3d}w  step median {med:7.2f} ms  "
              f"first {steps[0] if steps else float('nan'):8.2f} ms  denoise {m.get('stages', {}).get('DenoisingStage', float('nan')):8.1f} ms  "
              f"e2e {dt:6.2f} s  peak {m.get('memory_snapshots', {}).get('after_denoising', {}).get('peak_reserved_mb', float('nan'))}", flush=True)
        if os.environ.get("MEMLOG"):
            import subprocess

            used = subprocess.run(["nvidia-smi", "-i", os.environ["MEMLOG"], "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip()
            print(f"[mem] after req {len(out)}: {used} MiB", flush=True)
        out.append(dict(res=res, prompt=len(p.split()), steps=steps, e2e=dt))
    json.dump(out, open(os.environ.get("OUT", "/tmp/bcg_bench.json"), "w"))
    gen.shutdown()


if __name__ == "__main__":
    main()
