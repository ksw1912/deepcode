import torch
import numpy as np
import random

def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)  # 멀티 GPU를 사용하는 경우
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False