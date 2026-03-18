import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.rnn import RNN

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
plt.close("all")


#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()

#%%
def test(model, test_loader, device=None):
    model.eval()
    correct = 0
    total   = len(test_loader.dataset)
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.to(device), target.to(device)

            output = model(data)
            pred   = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().tolist())

    acc = 100.0 * correct / total
    return acc, all_preds


#%%
def main(args):
    # DataLoader 준비 - RNN을 위한 reshape
    mnist = np.load('mnist.npz')
    x_train = mnist['x_train']
    x_test  = (mnist['x_test']  - np.mean(x_train)) / np.std(x_train)
    y_test  = mnist['y_test']
    
    # 원본 이미지 보관 (시각화용)
    x_test_raw = mnist['x_test']
    
    # RNN을 위한 reshape: (N, 28, 28)
    x_test_rnn = x_test.reshape(-1, 28, 28)
    
    test_ds = TensorDataset(torch.FloatTensor(x_test_rnn), torch.LongTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 모델 경로 조합
    model_path = os.path.join(args.target_model_dir, f"epoch_{args.target_epoch}.pth")
    print(f"Loading model from: {model_path}")

    # 모델 초기화 및 가중치 로드
    device = torch.device(args.device) if args.device else torch.device('cpu')
    
    model = RNN(dimension=args.dimension,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=args.num_classes,
                device=device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # 테스트 수행
    test_acc, y_pred = test(model, test_loader, device=device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    # 샘플 시각화
    show_sample(x_test_raw, y_pred, y_test, idx=args.sample_idx)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained RNN model on MNIST")
    
    # Model hyperparameters (should match training)
    parser.add_argument("--dimension",     type=int,   default=28,     help="input dimension per time step")
    parser.add_argument("--hidden_size",   type=int,   default=64,    help="hidden size of RNN")
    parser.add_argument("--num_layers",    type=int,   default=2,      help="number of RNN layers")
    parser.add_argument("--num_classes",   type=int,   default=10,     help="number of output classes")
    
    # Test parameters
    parser.add_argument("--target_model_dir", type=str,   default="trained_output/rnn_mnist_b32_lr0.001_h64_l2"      )
    parser.add_argument("--target_epoch",     type=int,   default=7,     help="Epoch number of the model to test")
    parser.add_argument("--batch_size",       type=int,   default=64,     help="mini-batch size for testing")
    parser.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu" )
    parser.add_argument("--sample_idx",       type=int,   default=0,      help="index of the test sample to visualize")
    
    args = parser.parse_args()
    
    main(args)
