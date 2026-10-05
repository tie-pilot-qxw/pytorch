import torch, time, os
print("pid", os.getpid(), flush=True)
time.sleep(4)
print("props:", torch.cuda.get_device_properties(0).name, "init:", torch.cuda.is_initialized(), flush=True)
time.sleep(10)
