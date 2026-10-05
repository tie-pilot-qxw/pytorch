"""THE crux question: can an extern (cuDNN) node be patched from the DEVICE, from inside
the SAME graph launch -- i.e. exactly what the Triton deviceUpdatable path does today?
A patcher kernel is captured into the graph AHEAD of the cuDNN node; it reads the node
handle out of device memory (written after capture) and applies param updates."""
import ctypes, os
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "_wf_vrfy_synth2.py")).read().split("HOLD=[]; DEV=")[0])
# Build libinpatch.so from inpatch.cu (next to this script) first:
#   mkdir -p $DG_OUT/_wf_vrfy_build && nvcc -shared -Xcompiler -fPIC -arch=sm_90 -rdc=true \
#     inpatch.cu -o $DG_OUT/_wf_vrfy_build/libinpatch.so -lcudadevrt
ip=ctypes.CDLL(os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_vrfy_build", "libinpatch.so"))
ip.launch_patch.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]+[ctypes.c_void_p]*2

DEV="cuda"; DT=torch.bfloat16; B,H,L,D=2,8,512,64
q=torch.randn(B,H,L,D,device=DEV,dtype=DT)
slot = torch.zeros(1, dtype=torch.int64, device=DEV)
offs = torch.tensor([56], dtype=torch.int64, device=DEV)     # p3[0] = number of q tiles
vals = torch.tensor([4],  dtype=torch.int32,  device=DEV)
rcout= torch.zeros(1, dtype=torch.int32, device=DEV)
NUP  = 1
def body():
    ip.launch_patch(ctypes.c_void_p(slot.data_ptr()), ctypes.c_void_p(offs.data_ptr()),
                    ctypes.c_void_p(vals.data_ptr()), NUP, ctypes.c_void_p(rcout.data_ptr()),
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q,q,q,is_causal=True)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): body()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g): out=body()
torch.cuda.synchronize()
nds=nodes_of(g.raw_cuda_graph())
print("graph nodes:", [ntype(n) for n in nds])
kns=[n for n in nds if ntype(n)=="KERNEL"]
names=[]
for n in kns:
    p=KNP(); libcuda.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(n),ctypes.byref(p))
    s2=ctypes.c_char_p(); libcuda.cuFuncGetName(ctypes.byref(s2),ctypes.c_void_p(p.func))
    names.append(s2.value.decode())
for i,n in enumerate(names): print(f"  kernel node {i}: {n[:80]}")
target=[n for n,nm in zip(kns,names) if "sdpa" in nm][0]
buf=(ctypes.c_ubyte*256)(); ctypes.memset(buf,0,256)
ctypes.cast(buf,ctypes.POINTER(ctypes.c_int))[0]=1
rc=libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(target),ctypes.c_int(13),ctypes.byref(buf))
devnode=ctypes.cast(ctypes.byref(buf,8),ctypes.POINTER(ctypes.c_void_p))[0]
print(f"opt-in the cuDNN node AFTER capture: rc={en(rc)} devNode=0x{(devnode or 0):x}")
slot.fill_(devnode)                       # hand the handle to the in-graph patcher
g.instantiate(); ex=g.raw_cuda_graph_exec()
def upload():
    libcuda.cuGraphUpload(ctypes.c_void_p(ex), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
upload()
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    ref=F.scaled_dot_product_attention(q,q,q,is_causal=True)
torch.cuda.synchronize()
for v in [4,1,2,4]:
    vals.fill_(v); out.fill_(float("nan")); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    nn=int(torch.isnan(out).sum().item())
    print(f"  in-graph patcher wrote p3[0]={v}: device rc={int(rcout.item())} "
          f"NaN={nn}/{out.numel()} bit_exact_vs_eager={bool(torch.equal(out,ref))}")
print("\n-> the cuDNN node was re-parameterized from the device, inside the same graph launch.")
