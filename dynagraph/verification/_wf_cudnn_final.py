import subprocess, os, glob
# 1) offsets of the mem-eff AttentionKernel::Params fields (source-derived), vs measured diff
fields = [("query_ptr",8,8),("key_ptr",8,8),("value_ptr",8,8),("attn_bias_ptr",8,8),
          ("seqstart_q_ptr",8,8),("seqstart_k_ptr",8,8),("seqlen_k_ptr",8,8),
          ("causal_diagonal_offset",4,4),("output_ptr",8,8),("output_accum_ptr",8,8),
          ("logsumexp_ptr",8,8),("window_size",4,4),("scale",4,4),("head_dim",4,4),
          ("head_dim_value",4,4),("num_queries",4,4),("num_keys",4,4),
          ("num_keys_absolute",4,4),("custom_mask_type",1,1)]
off=0; layout={}
for n,sz,al in fields:
    off = (off + al - 1)//al*al
    layout[n]=off; off += sz
print("mem-eff AttentionKernel::Params, offsets from the PyTorch header:")
for n in ("output_ptr","num_queries","num_keys","head_dim"):
    print(f"   {n:24s} @ {layout[n]}")
print("   measured diff L=512->1024 hit [64:68] and [104:112];"
      f" output_ptr@{layout['output_ptr']}, num_queries@{layout['num_queries']},"
      f" num_keys@{layout['num_keys']}")
print("   -> match:", layout['output_ptr']==64 and layout['num_queries']==104 and layout['num_keys']==108)

# 2) is the cuDNN SDPA kernel present anywhere on disk?
print("\nsearching every nvidia/cudnn/cublas .so for the cuDNN SDPA kernel symbol:")
pats = ["/usr/lib/x86_64-linux-gnu/libcudnn*", "/usr/local/cuda/lib64/lib*",
        "/usr/local/lib/python3.12/dist-packages/nvidia/*/lib/*.so*"]
libs=[l for p in pats for l in glob.glob(p) if os.path.isfile(l) and os.path.getsize(l)>200000]
print(f"  {len(libs)} libraries scanned")
for needle in ["cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x64_4x1x1_cga1x1x1_kernel0_0",
               "fort_native_sdpa", "flash_fprop_wgmma"]:
    hits=[os.path.basename(l) for l in libs
          if subprocess.run(["grep","-q","-a","-F",needle,l]).returncode==0]
    print(f"  '{needle[:60]}...' -> {hits[:6] if hits else 'NOT PRESENT ON DISK'}")

# 3) any env knob in flash_api.cpp?
print("\nenv knobs in flash_api.cpp:")
r=subprocess.run(["grep","-n","getenv\\|Env\\|num_splits*=","/workspace/pytorch-main/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp"],capture_output=True,text=True)
print("  " + ("\n  ".join(r.stdout.strip().splitlines()[:10]) or "(none)"))
