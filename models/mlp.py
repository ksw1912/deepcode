import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,     256)
        self.fc3 = nn.Linear(256,     10)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.fc1(x)        
        x1 = self.bn1(x1)       
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)       
        
        x2 = self.fc2(x1)        
        x2 = self.bn2(x2)       
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)       
                
        x3 = self.fc3(x2)
        return x3

#%%
if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 8
    in_dim = 28*28  # 예: 28x28 이미지를 펼친 크기
    x = torch.randn(batch_size, in_dim)
    model = MLP(in_dim)
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # 모델 파라미터 및 연산량(MACs) 계산 pip install thop
    from thop import profile
    macs, params = profile(model, inputs=(x, ))
    print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    # torch.Size([8, 784]) -> torch.Size([8, 10])
    
    
    