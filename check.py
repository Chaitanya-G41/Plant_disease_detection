import torch

ckpt1 = torch.load('models/stage1/stage1_best.pth', map_location='cpu')
print("Stage 1 loaded")

ckpt2 = torch.load('models/stage2/stage2_best.pth', map_location='cpu')
print("Stage 2 loaded")