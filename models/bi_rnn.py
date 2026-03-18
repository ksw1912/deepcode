import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, dimension, hidden_size, num_layers, num_classes, device):
        super().__init__()
        
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device
        
        # if batch_first == True, then x is (batch_size, sequence_length, dimension)
        # otherwise, x is (sequence_length, batch_size, dimension)
        self.rnn = nn.RNN(self.dimension, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)
    
    def forward(self, x):
        # (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        
        # RNN forward: out shape = (batch_size, sequence_length, hidden_size * 2) for bidirectional
        out, h_out = self.rnn(x, h0)
        
        # Use the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

#%%
if __name__ == "__main__":
    # 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    sequence_length = 28
    dimension = 28
    hidden_size = 128
    num_layers = 1
    num_classes = 10
    
    # (batch_size, sequence_length, dimension)
    x = torch.randn(batch_size, sequence_length, dimension).to(device)
    
    model = BiRNN(dimension=dimension, 
                  hidden_size=hidden_size, 
                  num_layers=num_layers, 
                  num_classes=num_classes, 
                  device=device).to(device)
    
    output = model(x)
    
    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
