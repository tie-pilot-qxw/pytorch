"""Capture a CUDA graph around each extern op at two shapes; dump node list via
cudaGraphDebugDotPrint (torch.cuda.CUDAGraph.debug_dump). Counts nodes + kernel names."""
import os, re, glob, torch
dev = "cuda"
OUT = os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_externsurv_dots")
os.makedirs(OUT, exist_ok=True)


def cap(tag, fn):
    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        fn()
    p = f"{OUT}/{tag}.dot"
    g.debug_dump(p)
    txt = open(p).read()
    # node labels look like:  "graph_x_node_y"[style="..." label="{...}"]
    labels = re.findall(r'label="\{?([^"]*?)\}?"', txt)
    kern = []
    for l in labels:
        l = l.strip()
        if not l or l.startswith("graph") or l.startswith("ID "):
            continue
        kern.append(l.replace("\\n", " | ")[:160])
    print(f"\n### {tag}   (dot nodes with labels: {len(kern)})")
    for k in kern:
        print("    ", k)
    del g
    torch.cuda.synchronize()

bf = torch.bfloat16
for M in (128, 8192):
    a = torch.randn(M, 256, device=dev, dtype=bf); b = torch.randn(256, 256, device=dev, dtype=bf)
    o = torch.empty(M, 256, device=dev, dtype=bf)
    cap(f"mm_bf16_M{M}", lambda: torch.mm(a, b, out=o))

for M in (128, 8192):
    a = torch.randn(M, 256, device=dev, dtype=bf); b = torch.randn(256, 256, device=dev, dtype=bf)
    bias = torch.randn(256, device=dev, dtype=bf); o = torch.empty(M, 256, device=dev, dtype=bf)
    cap(f"addmm_bf16_M{M}", lambda: torch.addmm(bias, a, b, alpha=1, beta=1, out=o))

for H in (32, 64):
    x = torch.randn(4, 32, H, H, device=dev); w = torch.randn(64, 32, 3, 3, device=dev)
    cap(f"conv_fp32_H{H}", lambda: torch.convolution(x, w, None, (1,1), (1,1), (1,1), False, (0,0), 1))

for S in (128, 512):
    q = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    k = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    v = torch.randn(2, 4, S, 64, device=dev, dtype=bf)
    cap(f"sdpa_cudnn_S{S}", lambda: torch.ops.aten._scaled_dot_product_cudnn_attention.default(q, k, v, None, False))
    cap(f"sdpa_flash_S{S}", lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v))

for N in (64, 4096):
    x = torch.randn(8, N, device=dev)
    cap(f"sort_N{N}", lambda: torch.ops.aten.sort.stable(x, stable=False, dim=1, descending=False))
print("\nDONE")
