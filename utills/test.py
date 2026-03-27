import torch

ckpt = torch.load(r"C:\Users\KimSeowon\Desktop\git\deepcode\trained_output\FECDNet_fakeV_b8_lr0.0001\best_checkpoint_epoch_18.pth", map_location="cpu")

for k, v in ckpt.items():
    print(f"{k}: {v}")