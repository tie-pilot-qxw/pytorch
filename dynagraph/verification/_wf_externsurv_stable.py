"""Is the CUfunction handle in a captured node stable across repeated captures of the
SAME shape, and does it change across shapes? Distinguishes 'shape picks kernel' from
'module reload noise'."""
import os, re, torch
dev="cuda"; OUT=os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_externsurv_dots2"); os.makedirs(OUT, exist_ok=True)

def funcs(tag, fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); g.enable_debug_mode()
    with torch.cuda.graph(g): fn()
    p=f"{OUT}/{tag}.dot"; g.debug_dump(p); txt=open(p).read()
    fh = re.findall(r'func handle\} \| \{0x[0-9A-Fa-f]+ \| (0x[0-9A-Fa-f]+)', txt)
    names = re.findall(r'topoId: \d+\) \| ([A-Za-z_][\w:<>,\. ]*?)\\<\\<\\<', txt)
    del g; torch.cuda.synchronize()
    return fh, [n[:60] for n in names]

bf=torch.bfloat16
res={}
for rep in range(2):
    for S in (128, 512, 128):
        q=torch.randn(2,4,S,64,device=dev,dtype=bf); k=torch.randn(2,4,S,64,device=dev,dtype=bf); v=torch.randn(2,4,S,64,device=dev,dtype=bf)
        fh,nm = funcs(f"sdpa_S{S}_r{rep}", lambda: torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False))
        print(f"sdpa_cudnn S={S:4d} rep={rep}  func_handles={fh}  names={nm}")
    for M in (128, 8192, 128):
        a=torch.randn(M,256,device=dev,dtype=bf); b=torch.randn(256,256,device=dev,dtype=bf); o=torch.empty(M,256,device=dev,dtype=bf)
        fh,nm = funcs(f"mm_M{M}_r{rep}", lambda: torch.mm(a,b,out=o))
        print(f"mm bf16   M={M:4d} rep={rep}  func_handles={fh}  names={nm}")
print("DONE")
