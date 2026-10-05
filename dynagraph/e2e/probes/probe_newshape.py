import os, sys, time, torch
import transformers as t
torch.manual_seed(0)
attn = os.environ.get("ATTN", "sdpa")
m = t.BertForMaskedLM(t.BertConfig(attn_implementation=attn)).cuda().eval()
def run(L):
    x = torch.randint(5, 30000, (1, L), device="cuda"); mk = torch.ones_like(x); mk[:, L//2+1:] = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(input_ids=x, attention_mask=mk)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) * 1e3
run(64)
print("new ", [round(run(L), 1) for L in (100, 137, 211, 333)])
print("seen", [round(run(L), 1) for L in (100, 137, 211, 333)])
if os.environ.get("PROF"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        run(401)
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=15))
