import torch, sys
torch.cuda.init()
def free(): return torch.cuda.mem_get_info()[0]
f0 = free(); print("baseline free %.1f MiB" % (f0/2**20))
import triton
f1 = free(); print("after import triton: delta %.1f MiB" % ((f0-f1)/2**20))
from triton.runtime import driver
u = driver.active.utils
f2 = free(); print("after triton driver utils: delta %.1f MiB" % ((f1-f2)/2**20))
props = torch.cuda.get_device_properties(0)
f3 = free(); print("after get_device_properties: delta %.1f MiB (%s, sms=%d)" % ((f2-f3)/2**20, props.name, props.multi_processor_count))
import torch._inductor.runtime.hints as H
dp = H.DeviceProperties.create(torch.device("cuda:0"))
f4 = free(); print("after DeviceProperties.create: delta %.1f MiB -> %s" % ((f3-f4)/2**20, dp))
