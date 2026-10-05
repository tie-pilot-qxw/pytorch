#!/usr/bin/env python3
"""Does ExecUpdate wipe out node params that have already been patched?

This is the crux of "can ExecUpdate be used as a SWITCH". On every call DynaGraph
patches pointers and scalars on the exec according to the shape; if ExecUpdate re-parameterizes
every node from the template, then it is not on the same level as SWITCH --
every branch switch would mean redoing all the patches.

Method: capture a graph of y0 = x + 1, instantiate it, then use SetParams to point the output at z,
replay to confirm it writes z; then ExecUpdate back to the same template graph and replay to see where it writes.
"""
import ctypes
import torch
from cuda.bindings import runtime as cr, driver as cd
from torch.cuda._utils import _check_cuda_bindings as ck

x = torch.ones(1024, device="cuda")
y = torch.zeros(1024, device="cuda")
z = torch.zeros(1024, device="cuda")

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        torch.add(x, 1.0, out=y)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g):
    torch.add(x, 1.0, out=y)
torch.cuda.synchronize()
raw = g.raw_cuda_graph()
g.replay(); torch.cuda.synchronize()   # the exec only exists after instantiate
ex = g.raw_cuda_graph_exec()

nn = ck(cr.cudaGraphGetNodes(raw))[1]
nodes = ck(cr.cudaGraphGetNodes(raw, nn))[0]
node = nodes[0]
dp = ck(cd.cuGraphKernelNodeGetParams(node))

npar = 0
while True:
    try: ck(cd.cuFuncGetParamInfo(dp.func, npar))
    except RuntimeError: break
    npar += 1
    if npar > 64: break
arr = (ctypes.c_void_p * npar).from_address(int(dp.kernelParams))
blobs = []
for i in range(npar):
    _, sz = ck(cd.cuFuncGetParamInfo(dp.func, i))
    blobs.append(bytearray((ctypes.c_ubyte * int(sz)).from_address(arr[i])))

# find the 8 bytes holding the output pointer and replace them with z
hit = None
for i, bl in enumerate(blobs):
    for off in range(0, len(bl) - 7, 8):
        if int.from_bytes(bl[off:off + 8], "little") == y.data_ptr():
            hit = (i, off); break
    if hit: break
print(f"kernel={ck(cd.cuFuncGetName(dp.func)).decode()[:60]}  {npar} params  "
      f"y pointer is in param {hit[0]} at offset {hit[1]}")
blobs[hit[0]][hit[1]:hit[1] + 8] = z.data_ptr().to_bytes(8, "little")

KEEP = []
def params_from(blobs):
    p = cd.CUDA_KERNEL_NODE_PARAMS()
    p.func, p.kern, p.ctx = dp.func, dp.kern, dp.ctx
    p.gridDimX, p.gridDimY, p.gridDimZ = dp.gridDimX, dp.gridDimY, dp.gridDimZ
    p.blockDimX, p.blockDimY, p.blockDimZ = dp.blockDimX, dp.blockDimY, dp.blockDimZ
    p.sharedMemBytes = dp.sharedMemBytes
    cbs = [(ctypes.c_ubyte * len(b)).from_buffer(b) for b in blobs]
    a = (ctypes.c_void_p * len(cbs))(*[ctypes.cast(c, ctypes.c_void_p) for c in cbs])
    KEEP.append((blobs, cbs, a))
    p.kernelParams = ctypes.addressof(a); p.extra = 0
    return p

def show(tag):
    print(f"  {tag:<34} y[0]={y[0].item():<6.1f} z[0]={z[0].item():.1f}")

y.zero_(); z.zero_(); g.replay(); torch.cuda.synchronize()
show("1) plain replay")

ck(cd.cuGraphExecKernelNodeSetParams(ex, node, params_from(blobs)))
y.zero_(); z.zero_(); g.replay(); torch.cuda.synchronize()
show("2) SetParams points output at z")

rc = cr.cudaGraphExecUpdate(ex, raw)
print(f"  ExecUpdate(same template graph) rc={int(rc[0])}")
y.zero_(); z.zero_(); g.replay(); torch.cuda.synchronize()
show("3) replay after ExecUpdate")

print()
if y[0].item() == 2.0:
    print("  ==> ExecUpdate wiped the patch: the params were overwritten wholesale by the template graph.")
else:
    print("  ==> the patch survived: ExecUpdate did not touch this node.")
