import argparse
import torch
import os
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from data_loader import load_mnist
from models.rnn import RNN
from models.lstm import LSTM
from models.gru import GRU
import numpy as np
from models.gru1 import GRU_Attention


#%%
def train(model, train_loader, val_loader, criterion, optimizer, epochs: int=10, device=None, save_dir: str = None,patience: int = 5):
    model.train()
    total_train_samples = len(train_loader.dataset)
    total_val_samples   = len(val_loader.dataset)
    best_val_loss = float('inf')
    
    # Early Stopping 변수
    early_stop = False
    counter = 0

    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        # ----- Training -----
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0

        for data, target in train_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = train_loss_sum / len(train_loader)
        train_acc  = 100.0 * train_correct / total_train_samples

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        with torch.no_grad():
            for data, target in val_loader:
                if device is not None:
                    data, target = data.to(device), target.to(device)

                output = model(data)
                loss   = criterion(output, target)
                val_loss_sum += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc  = 100.0 * val_correct / total_val_samples

        # Print metrics for each epoch
        print(f"[Epoch {epoch:2d}] "
              f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        
        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            counter = 0 
            best_val_loss = val_loss
            if save_dir:
                file_name = f"epoch_{epoch}.pth"
                best_model_path = os.path.join(save_dir, file_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved to {best_model_path}")
        else:
            counter += 1
            print(f"  -> Validation loss did not improve. (counter: {counter}/{patience})")

            if counter >= patience:
                early_stop = True
                print("  -> Early stopping condition met!")
            

#%%
def main(args):
    # 1) DataLoader 준비 - RNN을 위해 데이터를 (batch, sequence_length, dimension)으로 reshape
    mnist = np.load('mnist.npz')
    x_train = (mnist['x_train'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
    y_train = mnist['y_train']
    
    # RNN을 위한 reshape: (N, 28, 28) - 각 행을 sequence의 time step으로 간주
    x_train_rnn = x_train.reshape(-1, 28, 28)  # (N, sequence_length=28, dimension=28)
    
    # Train/Validation split
    from torch.utils.data import TensorDataset
    full_train_ds = TensorDataset(torch.FloatTensor(x_train_rnn), torch.LongTensor(y_train))
    
    total_train = len(full_train_ds)
    val_size = int(args.val_portion * total_train)
    train_size = total_train - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_train_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None
    model = GRU_Attention(dimension=args.dimension, 
                hidden_size=args.hidden_size, 
                num_layers=args.num_layers, 
                num_classes=args.num_classes,
                device=device)

    if device is not None:
        model.to(device)
    
    criterion = CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 3) 저장 경로 설정
    save_dir = os.path.join(
        args.save_dir,
        f"gru_mnist_b{args.batch_size}_lr{args.lr}_h{args.hidden_size}_l{args.num_layers}"
    )
    
    # 4) 학습
    print(f"Training on {device}")
    print(f"Hyperparameters: batch_size={args.batch_size}, lr={args.lr}, "
          f"hidden_size={args.hidden_size}, num_layers={args.num_layers}, epochs={args.epochs}")
    print(f"Models will be saved to: {save_dir}")
    
    train(model, train_loader, val_loader, criterion, optimizer, 
          epochs=args.epochs, device=device, save_dir=save_dir)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN on MNIST")
    
    # Model hyperparameters
    parser.add_argument("--dimension",     type=int,   default=28,     help="input dimension per time step")
    parser.add_argument("--hidden_size",   type=int,   default=128,    help="hidden size of RNN")
    parser.add_argument("--num_layers",    type=int,   default=1,      help="number of RNN layers")
    parser.add_argument("--num_classes",   type=int,   default=10,     help="number of output classes")
    
    # Training parameters
    parser.add_argument("--batch_size",    type=int,   default=32,     help="mini-batch size")
    parser.add_argument("--lr",            type=float, default=1e-3,   help="learning rate")
    parser.add_argument("--epochs",        type=int,   default=100,     help="number of epochs to train")
    
    # Data parameters
    parser.add_argument("--val_portion",   type=float, default=0.1,    help="validation set portion")
    parser.add_argument("--shuffle",       type=bool,  default=True,   help="shuffle training data")
    
    # Device and saving
    parser.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for training")
    parser.add_argument("--save_dir",      type=str,   default="trained_output", 
                        help="directory to save trained models")
    
    args = parser.parse_args()
    
    main(args)
