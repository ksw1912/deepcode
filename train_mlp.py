import argparse
import torch
import os
from torch.optim    import SGD, Adagrad, RMSprop, Adam
from torch.nn       import CrossEntropyLoss
from torch.utils.data import  DataLoader
from data_loader import load_mnist
from models.mlp       import MLP


#%%
def train(model, train_loader, val_loader, criterion, optimizer, epochs: int=10, device=None, save_dir: str = None):
    model.train()
    total_train_samples = len(train_loader.dataset)
    total_val_samples   = len(val_loader.dataset)
    best_val_loss = float('inf')

    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
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
            best_val_loss = val_loss
            if save_dir:
                file_name = f"epoch_{epoch}.pth"
                best_model_path = os.path.join(save_dir, file_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved to {best_model_path}")
#%%
def main(args):
    # 1) DataLoader 준비
    train_ds, val_ds, _, in_dim, x_test_raw, y_test = load_mnist(
        use_small=args.use_small,
        valDB_portion = 0.1
    )
    train_loader = DataLoader(train_ds,  batch_size=args.batch_size,   shuffle=not args.no_shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,   shuffle=False)
    # test_loader  = DataLoader(test_ds,   batch_size=args.batch_size,   shuffle=False)
    
    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None
    model  = MLP(in_dim)

    if device is not None:
        model.to(device)
    criterion = CrossEntropyLoss()
    
########################################################################################    
    # optimizer = SGD(model.parameters(), lr=args.lr)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = Adagrad(model.parameters(), lr=args.lr)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.99)
    optimizer = Adam(model.parameters(), lr=args.lr)
########################################################################################        
    # 3) 학습
    
    # Construct save directory path
    save_dir = None
    if args.output_name:
        dir_name = f"{args.output_name}_b{args.batch_size}_lr{args.lr}"
        save_dir = os.path.join("trained_output", dir_name)

    train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=args.epochs,
                device=device,
                save_dir=save_dir
            )

    # 4) 학습된 모델 저장 (Handled in train function)
    # torch.save(model.state_dict(), args.save_path)
    # print(f"Model saved to {args.save_path}")
    print("Training finished. Best models were saved during training.")
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs",      type=int,   default=10,    help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=32,    help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-2,  help="learning rate for optimizer")
    parser.add_argument("--use_small",   type=bool,  default=False,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default=None,  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_name", type=str,   default="mlp_mnist", help="base name for the output directory")

    args = parser.parse_args()

    main(args)
