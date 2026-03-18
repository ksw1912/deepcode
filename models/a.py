import torch
import matplotlib.pyplot as plt

from gru1 import GRU_Attention
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 모델 초기화 (학습 시 사용했던 하이퍼파라미터 그대로!) ---
model = GRU_Attention(
    dimension=28,       # MNIST 입력 한 줄 28픽셀
    hidden_size=128,    # checkpoint에서 사용됨
    num_layers=1,
    num_classes=10,     # MNIST 클래스 개수
    device=device,
    bidirectional=True
).to(device)


import numpy as np
import torch
from torch.utils.data import TensorDataset
from data_loader import load_mnist
from torch.utils.data import DataLoader
# --- 학습된 가중치 로드 ---
model.load_state_dict(torch.load(r"E:\2025_deeplearning\pr_week_13_torch_mnist_LSTM\trained_output\lstm_mnist_b32_lr0.001_h128_l1\epoch_6.pth", map_location=device))
model.eval()

  _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist( use_small=False, valDB_portion = 0.1 )
    test_loader  = DataLoader(test_ds,   batch_size=32,   shuffle=False)

# --- 테스트용 입력 (실제 입력으로 바꿔도 됨) ---
x = torch.randn(1, 28, 28).to(device)  # (B, L, dimension)

# --- 모델 실행 ---
with torch.no_grad():
    output, attn_weight = model(x)


att = attn_weight[0].cpu().numpy()  # shape: (28,)
# 첫 번째 sample의 attention
plt.figure(figsize=(14, 3))

# 2D 형태로 reshape (height=1, width=28)
heatmap_data = att.reshape(1, -1)

plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
plt.colorbar(label="Attention Weight")
plt.title("Attention Heatmap (Matplotlib Only)")
plt.xlabel("Time Step (Sequence Index)")
plt.yticks([])  # Y축 제거

plt.show()