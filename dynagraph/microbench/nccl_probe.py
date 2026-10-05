"""Is NCCL 'yet another kernel'? Do the kernel / grid / topology change when the message size changes?"""
import os, sys, torch, torch.distributed as dist
import cuda.bindings.driver as D

def chk(r):
    if isinstance(r,(list,tuple)):
        if r[0] != D.CUresult.CUDA_SUCCESS: raise RuntimeError(str(r[0]))
        rest = r[1:]; return rest[0] if len(rest)==1 else rest
    return r

def nodes_of(g):
    cg = D.CUgraph(g.raw_cuda_graph())
    _, cnt = chk(D.cuGraphGetNodes(cg, 0))
    nds, _ = chk(D.cuGraphGetNodes(cg, cnt))
    out=[]
    for nd in nds:
        t = chk(D.cuGraphNodeGetType(nd))
        if t != D.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            out.append((str(t).split(".")[-1], None)); continue
        p = chk(D.cuGraphKernelNodeGetParams(nd))
        try: nm = chk(D.cuFuncGetName(p.func)).decode()
        except Exception: nm = "?"
        out.append((nm[:56], (p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX)))
    return out

def main():
    rank = int(os.environ["RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    s = torch.cuda.Stream()
    results = {}
    for n in [1024, 65536, 1<<20, 1<<24]:
        x = torch.ones(n, device="cuda", dtype=torch.float16)
        for _ in range(3):
            with torch.cuda.stream(s): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(g, stream=s): dist.all_reduce(x)
            torch.cuda.synchronize()
            results[n] = nodes_of(g)
        except Exception as e:
            results[n] = [(f"capture failed: {type(e).__name__} {str(e)[:60]}", None)]
    if rank == 0:
        print(f"{'elements':>10}  {'nodes':>6}  kernel / grid")
        print("-"*92)
        sig = {}
        for n, nds in results.items():
            k = tuple(x[0] for x in nds)
            sig.setdefault(k, []).append(n)
            gs = " ".join(f"{x[1][0]}x{x[1][3]}" if x[1] else "-" for x in nds)
            print(f"{n:>10}  {len(nds):>6}  " + " | ".join(x[0][:34] for x in nds))
            print(f"{'':>10}  {'':>6}  grid x block: {gs}")
        print(f"\nAcross 4 sizes (1K -> 16M elements, 16384x): {len(sig)} distinct kernel sequences")
        allg = [tuple(x[1] for x in nds if x[1]) for nds in results.values()]
        print(f"grid all identical: {len(set(allg))==1}   -> {'NCCL is tier 1: only parameters (count/pointers) change' if len(sig)==1 and len(set(allg))==1 else 'something changes, see above'}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
