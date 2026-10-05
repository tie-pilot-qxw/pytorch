"""Finish it: device-side cudaGraphKernelNodeSetParam on the cuDNN SDPA node.
(The grid test was inconclusive because that cuDNN kernel is persistent: grid=1 still
does all the work.  A param write is unambiguous.)"""
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
dev.launch_set_grid.restype=ctypes.c_int
dev.launch_set_grid.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint]

c = cap(512)
kn, out, q = c['kn'], c['out'], c['q']
print(f"kernel grid={c['grid']} block={c['block']}, param_info={c['pi']}")
buf=(ctypes.c_ubyte*256)(); ctypes.memset(buf,0,256)
ctypes.cast(buf,ctypes.POINTER(ctypes.c_int))[0]=1
rc=libcuda.cuGraphKernelNodeSetAttribute(ctypes.c_void_p(kn),ctypes.c_int(13),ctypes.byref(buf))
devnode=ctypes.cast(ctypes.byref(buf,8),ctypes.POINTER(ctypes.c_void_p))[0]
print(f"opt-in on the cuDNN node: rc={en(rc)} devNode=0x{(devnode or 0):x}")
c['g'].instantiate(); ex=c['g'].raw_cuda_graph_exec()
libcuda.cuGraphUpload(ctypes.c_void_p(ex), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    ref=F.scaled_dot_product_attention(q,q,q,is_causal=True)
torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); c['g'].replay(); torch.cuda.synchronize()
print(f"baseline replay bit_exact={bool(torch.equal(out,ref))}")

# param0 lives at device-side offset 0; its [16:20] word is the seq length (=512)
r=dev.launch_set_param(ctypes.c_void_p(devnode), ctypes.c_size_t(16), ctypes.c_uint(256))
torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); c['g'].replay(); torch.cuda.synchronize()
changed = not bool(torch.equal(out,ref))
nn=int(torch.isnan(out).sum().item())
print(f"device-side SetParam(off=16, L:512->256) rc={r}: output changed={changed} NaN={nn}/{out.numel()}")

r2=dev.launch_set_param(ctypes.c_void_p(devnode), ctypes.c_size_t(16), ctypes.c_uint(512))
torch.cuda.synchronize()
out.fill_(float("nan")); torch.cuda.synchronize(); c['g'].replay(); torch.cuda.synchronize()
print(f"restore rc={r2}: bit_exact_vs_eager_again={bool(torch.equal(out,ref))}")
print(f"\n-> device-side param update reaches an EXTERN cuDNN node: {r==0 and changed and bool(torch.equal(out,ref))}")
