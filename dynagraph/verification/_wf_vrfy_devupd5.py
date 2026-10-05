"""Which device-side param offsets actually reach the cuDNN kernel?  Probe the shape
words already identified (all reductions, so no OOB reads), one at a time."""
import ctypes, os
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "_wf_vrfy_synth2.py")).read().split("FIT=[256,512,1024]")[0])
# Build libsetgrid.so from setgrid.cu (next to this script) first:
#   mkdir -p $DG_OUT/_wf_vrfy_build && nvcc -shared -Xcompiler -fPIC -arch=sm_90 -rdc=true \
#     setgrid.cu -o $DG_OUT/_wf_vrfy_build/libsetgrid.so -lcudadevrt
dev=ctypes.CDLL(os.path.join(os.environ.get("DG_OUT", "/tmp/dynagraph_out"), "_wf_vrfy_build", "libsetgrid.so"))
dev.launch_set_param.restype=ctypes.c_int
dev.launch_set_param.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint]
c=cap(512); kn,out,q=c['kn'],c['out'],c['q']
buf=(ctypes.c_ubyte*256)(); ctypes.memset(buf,0,256)
ctypes.cast(buf,ctypes.POINTER(ctypes.c_int))[0]=1
libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(kn),ctypes.c_int(13),ctypes.byref(buf))
devnode=ctypes.cast(ctypes.byref(buf,8),ctypes.POINTER(ctypes.c_void_p))[0]
c['g'].instantiate(); ex=c['g'].raw_cuda_graph_exec()
libcuda.cuGraphUpload(ctypes.c_void_p(ex), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    ref=F.scaled_dot_product_attention(q,q,q,is_causal=True)
torch.cuda.synchronize()
# flat device-side offsets: p0@0 p1@40 p2@44 p3@56 p5@112
probes=[("p0[16] seqlen_q",16,512,256),("p0[20] seqlen_kv",20,512,256),
        ("p1[0] total CTAs",40,64,16),("p2[0] ctas/2",44,32,8),
        ("p3[0] q-tiles",56,4,1),("p5[36] L-1",112+36,511,255)]
for name,off,old,new in probes:
    r=dev.launch_set_param(ctypes.c_void_p(devnode),ctypes.c_size_t(off),ctypes.c_uint(new))
    torch.cuda.synchronize()
    out.fill_(float("nan")); torch.cuda.synchronize(); c['g'].replay(); torch.cuda.synchronize()
    nn=int(torch.isnan(out).sum().item()); ch=not bool(torch.equal(out,ref))
    print(f"  off={off:4d} {name:18s} {old}->{new}: rc={r} output_changed={ch} NaN={nn}/{out.numel()}")
    dev.launch_set_param(ctypes.c_void_p(devnode),ctypes.c_size_t(off),ctypes.c_uint(old)); torch.cuda.synchronize()
    out.fill_(float("nan")); torch.cuda.synchronize(); c['g'].replay(); torch.cuda.synchronize()
    assert torch.equal(out,ref), f"restore failed after {name}"
print("  (every probe restored cleanly to the bit-exact baseline)")
