import os, sys, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_aten_build")
os.makedirs(_BUILD, exist_ok=True)

probe = load(
    name="_wf_aten_graphprobe",
    sources=[os.path.join(_HERE, "_wf_aten_graphprobe.cpp")],
    extra_ldflags=["-lcuda", "-L/usr/local/cuda/lib64/stubs", "-L/usr/local/cuda/lib64"],
    extra_cflags=["-O1", "-std=c++20", "-I/usr/local/cuda/include"],
    build_directory=_BUILD,
    verbose=False,
)

def capture(fn, warmup=3):
    """Warm up fn on a side stream, then capture exactly one call into a graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    nodes = probe.dump_graph(g.raw_cuda_graph())
    return g, nodes

def kernels(nodes):
    return [n for n in nodes if n["type"] == "KERNEL"]

def demangle(name):
    import subprocess
    try:
        return subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
    except Exception:
        return name

def short(name, n=110):
    d = demangle(name)
    return d if len(d) <= n else d[:n] + "..."

def pshow(p):
    b = p["bytes"]
    if b is None:
        return "?"
    if len(b) == 8:
        v = int.from_bytes(b, "little")
        return f"u64:{v}"
    if len(b) == 4:
        return f"u32:{int.from_bytes(b,'little')}"
    if len(b) in (1, 2):
        return f"u{len(b)*8}:{int.from_bytes(b,'little')}"
    return b.hex()

def diff_params(n1, n2):
    """Return list of (index, offset, size, bytes1, bytes2) for params that differ."""
    out = []
    for a, b in zip(n1["params"], n2["params"]):
        if a["bytes"] != b["bytes"]:
            out.append((a["index"], a["offset"], a["size"], a["bytes"], b["bytes"]))
    return out
