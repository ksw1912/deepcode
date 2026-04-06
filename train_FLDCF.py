import argparse
import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from dataloaders.data_loader_hrcus_fake import HRCUS_FAKE
from dataloaders.data_loader_fakeV import Fake_Vaihingen_LoveDA
from dataloaders.data_loader_fakeL import Fake_LoveDA
from models.fldcf_dir.fldcf import FLDCF
from loss.fldcf_loss.fldcf_loss import Loss_fake
from train.seed_setting import set_seed
from tqdm import tqdm
from utills.metrics import compute_seg_metrics, comfusion_matrix
from utills.history_info import HistoryManager, history_save


# %%
def train(model, train_loader, val_loader, criterion, optimizer, args, epochs: int = 10, device=None, save_dir: str = None):
    model.train()
    best_val_mIou = 0.0
    history = HistoryManager()

    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        train_loss_sum = 0.0
        total_train_samples = 0
        train_correct = 0
        train_tp, train_tn, train_fp, train_fn = 0.0, 0.0, 0.0, 0.0
        val_fn, val_tn, val_fp, val_tp = 0.0, 0.0, 0.0, 0.0
        train_sg_mf1, train_seg_miou, train_seg_oa = 0.0, 0.0, 0.0
        num_batches = 0.0

        train_pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}", leave=True)

        # for batch in train_loader:
        for batch in train_pbar:
            data = batch['image']
            gt_mask = batch['mask']
            cls_label = batch["is_fake"]

            if device is not None:
                data = data.to(device)
                gt_mask = gt_mask.to(device)
                cls_label = cls_label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion.loss_calc(
                out=output,
                label=gt_mask,
                out_label=cls_label,
                model='fldcf'
            ).to(device)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            seg_pred, cls_pred = output

            # segmentation
            pred_mask = seg_pred.argmax(dim=1)  # keep deep= False
            if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
                gt_mask = gt_mask.squeeze(1)

            tp, tn, fp, fn = comfusion_matrix(pred_mask, gt_mask)
            train_tp += tp
            train_tn += tn
            train_fp += fp
            train_fn += fn

            # classification용
            pred_cls = cls_pred.argmax(dim=1)  # [B]
            batch_correct = (pred_cls == cls_label).sum().item()  # or pred_cls.eq(cls_label).sum().item()

            num_batches += 1
            train_correct += batch_correct
            total_train_samples += cls_label.size(0)

        train_seg_mf1, train_seg_miou, train_seg_oa = compute_seg_metrics(train_tp, train_tn, train_fp, train_fn)
        train_loss = train_loss_sum / len(train_loader)
        train_cls_acc = 100.0 * train_correct / total_train_samples

        train_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            # classification
            train_acc=f"{train_cls_acc:.2f}%",
            # segmentation
            train_seg_mf1=f"{train_seg_mf1:.3f}",
            train_seg_iou=f"{train_seg_miou:.3f}",
            train_seg_oa=f"{train_seg_oa:.3f}"
        )

        # ----------------------------------------------------------------------------- ------------------------
        # ----- Validation ------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------- ------------------------
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        total_val_samples = 0

        val_seg_mf1 = 0.0
        val_seg_miou = 0.0
        val_seg_oa = 0.0

        val_batches = 0

        val_pbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch}/{epochs}", leave=True)

        with (torch.no_grad()):
            # for batch in train_loader:
            for batch in val_pbar:
                data = batch['image']
                gt_mask = batch['mask']
                cls_label = batch["is_fake"]

                if device is not None:
                    data = data.to(device)
                    gt_mask = gt_mask.to(device)
                    cls_label = cls_label.to(device)

                output = model(data)
                seg_pred, cls_pred = output

                print("=== BEFORE LOSS ===")
                print("seg_pred:", seg_pred.shape)
                print("gt_mask :", gt_mask.shape)
                print("gt_mask dtype:", gt_mask.dtype)
                print("gt_mask unique:", torch.unique(gt_mask))

                loss = criterion.loss_calc(
                    out=output,
                    label=gt_mask,
                    out_label=cls_label,
                    model='fldcf'
                )

                val_loss_sum += loss.item()


                # segmentation용
                pred_mask = seg_pred.argmax(dim=1)  # [B, H, W]

                if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
                    gt_mask = gt_mask.squeeze(1)

                tp, tn, fp, fn = comfusion_matrix(pred_mask, gt_mask)
                val_tp += tp
                val_tn += tn
                val_fp += fp
                val_fn += fn

                # classification용
                pred_cls = cls_pred.argmax(dim=1)  # [0,1] => [1]
                batch_correct = (pred_cls == cls_label).sum().item()

                val_batches += 1
                val_correct += batch_correct
                total_val_samples += cls_label.size(0)

        val_seg_mf1, val_seg_miou, val_seg_oa = compute_seg_metrics(val_tp, val_tn, val_fp, val_fn)
        val_loss = val_loss_sum / len(val_loader)
        val_cls_acc = 100.0 * val_correct / total_val_samples

        val_pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            av_acc=f"{val_cls_acc:.2f}%",
            f1=f"{val_seg_mf1:.3f}",
            miou=f"{val_seg_miou:.3f}",
            oa=f"{val_seg_oa:.3f}"
        )

        # Print metrics for each epoch
        print(
            f"[Epoch {epoch:2d}] "
            f"Train Loss: {train_loss:.6f}, Acc: {train_cls_acc:.2f}%, mF1: {train_seg_mf1:.3f}, mIoU: {train_seg_miou:.3f}, Overall accuracy: {train_seg_oa:.3f} | "
            f"Val Loss: {val_loss:.6f}, Acc: {val_cls_acc:.2f}%, mF1: {val_seg_mf1:.3f}, mIoU: {val_seg_miou:.3f}, Overall accuracy: {val_seg_oa:.3f}"
        )

        # Save the model if validation loss has improved
        if best_val_mIou < val_seg_miou:
            best_val_mIou = val_seg_miou
            if save_dir:


                file_name = f"epoch_{epoch}_trainF1_{train_seg_mf1:.4f}_trainmIOU_{train_seg_miou:.4f}.pth"
                best_model_path = os.path.join(save_dir, file_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved to {best_model_path}")


                best_history_path = os.path.join(save_dir, f"best_checkpoint_epoch_{epoch}.pth")

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_cls_acc": train_cls_acc,
                    "train_seg_mf1": train_seg_mf1,
                    "train_seg_miou": train_seg_miou,
                    "train_seg_oa": train_seg_oa,
                    "val_loss": val_loss,
                    "val_cls_acc": val_cls_acc,
                    "val_seg_mf1": val_seg_mf1,
                    "val_seg_miou": val_seg_miou,
                    "val_seg_oa": val_seg_oa
                }, best_history_path)

        # 2) last model 저장 (마지막 epoch에서만)
        if epoch == args.epochs:
            last_model_path = os.path.join(save_dir, "last_model.pth")
            torch.save(model.state_dict(), last_model_path)

            last_ckpt_path = os.path.join(save_dir, "last_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_cls_acc": train_cls_acc,
                "train_seg_mf1": train_seg_mf1,
                "train_seg_miou": train_seg_miou,
                "train_seg_oa": train_seg_oa,
                "val_loss": val_loss,
                "val_cls_acc": val_cls_acc,
                "val_seg_mf1": val_seg_mf1,
                "val_seg_miou": val_seg_miou,
                "val_seg_oa": val_seg_oa,
            }, last_ckpt_path)

            print(f"  -> Last model saved to {last_model_path}")
            print(f"  -> Last checkpoint saved to {last_ckpt_path}")

        # 3) 학습 정보 객체 값 c
        history.update_history(train_loss, train_cls_acc, train_seg_mf1, train_seg_miou, train_seg_oa, val_loss,
                               val_cls_acc,
                               val_seg_mf1, val_seg_miou, val_seg_oa)

    # 학습 과정 정보 저장
    history_root = os.path.join(save_dir, "history",args.dataset,args.output_name)
    history_save(history,path=history_root)


# %%
def main(args):
    # seed 고정
    set_seed(args.seed)
    # 1) DataLoader 준비
    if args.dataset == "HRCUS_FAKE":
        train_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16", split="train")  # dataset 경로
        val_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16", split="val")
        # test_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/test", split="test")

    elif args.dataset == "Fake-LoveDA":
        train_dataset_not_split = Fake_LoveDA(root_dir='./dataset/Fake-LoveDA', split="train")
        test_dataset = Fake_LoveDA('./dataset/Fake-LoveDA', split="test")

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
        train_dataset_not_split = Fake_Vaihingen_LoveDA(root_dir='./dataset/Fake-Vaihingen', split="train")
        test_dataset = Fake_Vaihingen_LoveDA('./dataset/Fake-Vaihingen', split="test")

        train_size = int(0.8 * len(train_dataset_not_split))
        val_size = len(train_dataset_not_split) - train_size

        train_dataset, val_dataset = random_split(
            train_dataset_not_split,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

    else:
        print("만들면 채우기")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,drop_last =True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,drop_last =True)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None

    model = FLDCF(args)

    if device is not None:
        model.to(device)
    criterion = Loss_fake()

    ########################################################################################
    # optimizer = SGD(model.parameters(), lr=args.lr)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = Adagrad(model.parameters(), lr=args.lr)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.99)
    optimizer = AdamW(model.parameters(), betas=(0.99, 0.999), lr=args.lr, weight_decay=0, eps=1e-8)
    ########################################################################################
    # 3) 학습

    # Construct save directory path
    # 3) 저장 경로 설정
    save_dir = os.path.join(
        args.save_dir,
        f"{args.output_name}_b{args.batch_size}_lr{args.lr}"
    )

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args,
        epochs=args.epochs,
        device=device,
        save_dir=save_dir,
    )


# %%
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="remote-sensing")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="mini-batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for optimizer")
    # parser.add_argument("--imageSize",   type=int,  default=256,  help="input_size")
    parser.add_argument("--shuffle", type=bool, default=True, help="disable data shuffling")
    parser.add_argument("--device", type=str, default='cuda', help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_name", type=str, default="fldcf_fakeL1", help="base name for the output directory")
    # dataset 변경시 -> dataset 경로도 추가로 지정 필요
    parser.add_argument("--dataset", type=str, default='Fake-LoveDA',
                        help="dataset setting(HRCUS_FAKE,Fake-LoveDA,Fake-Vaihingen,Splice-Vaihingen)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--save_dir", type=str, default="trained_output",
                        help="directory to save trained models")

    # model 변경 -> 자동으로 #loss함수 바뀌게 세팅 필요

    args = parser.parse_args()

    main(args)
