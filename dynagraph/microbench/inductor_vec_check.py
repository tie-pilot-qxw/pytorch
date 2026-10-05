import torch, os, glob, re
torch._dynamo.config.cache_size_limit = 64
@torch.compile(dynamic=True)
def f(x, y): return x * y + 1.0
for n in [777, 4096, 100003]:
    x = torch.randn(n, device="cuda"); y = torch.randn_like(x); f(x, y)
torch.cuda.synchronize()
seen = set()
for p in glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/*.ptx", recursive=True):
    src = open(p).read(); name = os.path.basename(p)
    ents = re.findall(r"\.entry\s+(\w+)", src)
    print(f"{name:40s} v4 loads={src.count('ld.global.v4'):2d}  scalar32 loads={src.count('ld.global.f32')+src.count('ld.global.b32'):2d}")
