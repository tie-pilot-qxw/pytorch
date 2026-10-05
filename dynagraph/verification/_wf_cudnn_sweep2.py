"""Topology sweep, releasing each graph so the private pools don't accumulate."""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from _wf_cudnn_lib import graph_nodes, node_type, kernel_params, func_name

def sweep(label, cases, fnmaker):
    print(f"\n{'='*92}\n### {label}")
    topo={}
    for c in cases:
        fn = fnmaker(c)
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(g): out=fn()
            torch.cuda.synchronize()
            sg=[]
            for nd in graph_nodes(g.raw_cuda_graph()):
                t=node_type(nd)
                if t!="KERNEL": sg.append((t,None,None)); continue
                p=kernel_params(nd); sg.append((t,func_name(p.func),(p.gridDimX,p.gridDimY,p.gridDimZ)))
        except Exception as e:
            print(f"  {c}: FAIL {type(e).__name__}: {str(e)[:80]}"); continue
        finally:
            del out
            g.reset(); del g; gc.collect(); torch.cuda.empty_cache()
        key=tuple((t,n) for t,n,_ in sg)
        topo.setdefault(key,[]).append(c)
        print(f"  {c!s:>6}: {len(sg)} nodes grids={[x for _,_,x in sg]}")
    print(f"  -> {len(topo)} distinct topolog{'y' if len(topo)==1 else 'ies'} over {len(cases)} shapes:")
    for key,cs in topo.items():
        print(f"     {cs}")
        for t,n in key: print(f"        {t} {(n or '')[:95]}")
    return topo

DEV="cuda"
W=torch.randn(64,64,3,3,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
XB=torch.randn(48,64,56,56,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
sweep("conv2d channels_last, sweep batch", [1,2,3,4,5,6,7,8,12,16,20,24,32,48],
      lambda b: (lambda: F.conv2d(XB[:b],W,padding=1)))
XN=XB.contiguous()
WN=W.contiguous()
sweep("conv2d NCHW, sweep batch", [1,2,4,8,16,32],
      lambda b: (lambda: F.conv2d(XN[:b],WN,padding=1)))
# resolution sweep at fixed batch (the "variable image size" axis)
XR=torch.randn(8,64,128,128,device=DEV,dtype=torch.float16).to(memory_format=torch.channels_last)
sweep("conv2d channels_last, batch=8, sweep H=W", [32,48,56,64,96,112,128],
      lambda r: (lambda: F.conv2d(XR[:, :, :r, :r].contiguous(memory_format=torch.channels_last),W,padding=1)))
del XB, XN, XR; gc.collect(); torch.cuda.empty_cache()

QB=torch.randn(2,8,4096,64,device=DEV,dtype=torch.bfloat16)
def mk(bk):
    def f(L):
        def g():
            with sdpa_kernel(bk):
                return F.scaled_dot_product_attention(QB[:,:,:L],QB[:,:,:L],QB[:,:,:L],is_causal=True)
        return g
    return f
LS=[128,256,333,512,777,1024,1536,2048,3000,4096]
sweep("SDPA CUDNN, sweep seqlen", LS, mk(SDPBackend.CUDNN_ATTENTION))
sweep("SDPA FLASH, sweep seqlen", LS, mk(SDPBackend.FLASH_ATTENTION))
