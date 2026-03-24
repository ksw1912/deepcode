import argparse
import torch
import os

from sympy.physics.units import momentum
from torch.optim import SGD, Adagrad, RMSprop, Adam, AdamW
from torch.nn import L1Loss
from torch.utils.data import DataLoader, random_split
from dataloaders.data_loader_hrcus_fake import HRCUS_FAKE
from dataloaders.data_loader_fakeV import Fake_Vaihingen
from dataloaders.data_loader_fakeL import Fake_LoveDA
from models.fldcf_dir.Restore import Restoretest
from train.seed_setting import set_seed

# %%
def train(model, train_loader, val_loader, criterion, optimizer, epochs: int = 10, device=None, save_dir: str = None):
    model.train()
    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)
    best_val_loss = float('inf')
    patience = 0

    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0

        for batch in train_loader:
            if device is not None:
                data = batch["image"]
                target = batch["mask"]
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output, RDBs_out = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = train_loss_sum / len(train_loader)
        train_acc = 100.0 * train_correct / total_train_samples

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                if device is not None:
                    data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)
                val_loss_sum += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc = 100.0 * val_correct / total_val_samples

        # Print metrics for each epoch
        print(f"[Epoch {epoch:2d}] "
              f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")

        # Save the model if validation loss has improved
        # ----- Check improvement -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            if save_dir:
                file_name = f"best_epoch_{epoch}.pth"
                best_model_path = os.path.join(save_dir, file_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved to {best_model_path}")
        else:
            early_stop_counter += 1
            print(f"  -> No improvement. EarlyStopping counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
# %%
def main(args):
    #seed 고정
    set_seed(args.seed)

    # 1) DataLoader 준비
    if args.dataset == "HRCUS_FAKE":
        train_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/train", split="train")  # dataset 경로
        val_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/val", split="val")
        # test_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/test", split="test")

    elif args.dataset == "Fake-LoveDA":
        train_dataset_not_split = Fake_LoveDA(root_dir='../../dataset/Fake-LoveDA', split="train")
        test_dataset = Fake_LoveDA('../../dataset/Fake-LoveDA', split="test")

        train_size = int(0.8 * len(train_dataset_not_split))
        val_size = len(train_dataset_not_split) - train_size

        train_size = int(0.8 * len(train_dataset_not_split))
        val_size = len(train_dataset_not_split) - train_size

        train_dataset, val_dataset = random_split(
            train_dataset_not_split,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

    elif args.dataset == "Fake-Vaihingen":
        train_dataset_not_split = Fake_Vaihingen(root_dir='../dataset/Fake-Vaihingen', split="train")
        test_dataset = Fake_Vaihingen('../dataset/Fake-Vaihingen', split="test")

        train_size = int(0.8 * len(train_dataset_not_split))
        val_size = len(train_dataset_not_split) - train_size

        train_dataset, val_dataset = random_split(
            train_dataset_not_split,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

    else:
        print("만들면 채우기")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None

    # 추후 수정 필요!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model = Restoretest(args)

    if device is not None:
        model.to(device)
    criterion = L1Loss()

    ########################################################################################
    # optimizer = SGD(model.parameters(), lr=args.lr)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = Adagrad(model.parameters(), lr=args.lr)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.99)
    optimizer = AdamW(model.parameters(),betas=(0.99, 0.999), lr=args.lr, weight_decay=0, eps = 1e-8)
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



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="remote-sensing")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="mini-batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for optimizer")
    # parser.add_argument("--imageSize",   type=int,  default=256,  help="input_size")
    parser.add_argument("--no_shuffle", type=bool, default=True, help="disable data shuffling")
    parser.add_argument("--device", type=str, default='cuda', help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_name", type=str, default="fldcf_content_prior", help="base name for the output directory")
    # dataset 변경시 -> dataset 경로도 추가로 지정 필요
    parser.add_argument("--dataset", type=str, default='Fake-LoveDA',
                        help="dataset setting(HRCUS_FAKE,Fake-LoveDA,Fake-Vaihingen,Splice-Vaihingen)")
    parser.add_argument("--seed", type=int, default=42 , help="random seed for initialization")


    # model 변경 -> 자동으로 #loss함수 바뀌게 세팅 필요

    args = parser.parse_args()

    main(args)
