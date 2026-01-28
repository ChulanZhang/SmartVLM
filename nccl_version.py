import torch
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.nccl.version())
