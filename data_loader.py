import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_mnist(use_small: bool=True,  valDB_portion: float=0.0):
    # 1) MNIST npz 로드 및 정규화
    mnist   = np.load(r'E:\\2025_deeplearning\\pr_week_13_torch_mnist_LSTM\\mnist.npz')
    x_train = (mnist['x_train'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
    y_train = mnist['y_train']
    x_test  = (mnist['x_test']  - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
    y_test  = mnist['y_test']

    # 2) 다운사이징(7×7) 혹은 원본(28×28)
    if use_small:
        x_train_in = x_train[:, ::4, ::4].reshape(-1, 7*7)
        x_test_in  = x_test[:,  ::4, ::4].reshape(-1, 7*7)
    else:
        x_train_in = x_train.reshape(-1, 28*28)
        x_test_in  = x_test.reshape(-1, 28*28)

    input_dim = x_train_in.shape[1]

    # 3) TensorDataset & DataLoader
    train_ds = TensorDataset(torch.FloatTensor(x_train_in), torch.LongTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(x_test_in),  torch.LongTensor(y_test))

    # Train/Validation split 
    total_train = len(train_ds)
    val_size     = int(valDB_portion * total_train)
    train_size   = total_train - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # x_test 원본(28×28)과 y_test도 반환해서 시각화 등에 사용
    return train_ds, val_ds, test_ds, input_dim, x_test, y_test
