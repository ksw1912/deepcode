import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataloaders.data_loader_hrcus_fake import HRCUS_FAKE
from dataloaders.data_loader_fakeV import Fake_Vaihingen_LoveDA
from dataloaders.data_loader_fakeL import Fake_LoveDA
from models.fldcf_dir.fldcf import FLDCF
from train.seed_setting import set_seed
from utills.metrics import compute_seg_metrics, comfusion_matrix


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_overlay(image, gt_mask, pred_mask, alpha=0.45):
    """
    image:    [C,H,W]
    gt_mask:  [H,W] or [1,H,W]
    pred_mask:[H,W] or [1,H,W]

    TP: green   (맞춘 영역)
    FP: red     (과검출)
    FN: yellow  (놓친 영역)
    """
    img = image.detach().cpu().float().clone()

    # image range
    if img.max() > 1.0:
        img = img / 255.0

    # [1,H,W] -> [3,H,W]
    if img.dim() == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    elif img.dim() == 3 and img.size(0) == 1:
        img = img.repeat(3, 1, 1)

    gt = gt_mask.detach().cpu().squeeze().bool()     # [H,W]
    pred = pred_mask.detach().cpu().squeeze().bool() # [H,W]

    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    overlay = img.clone()

    green = torch.tensor([0.0, 1.0, 0.0], dtype=overlay.dtype).view(3, 1)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=overlay.dtype).view(3, 1)
    yellow = torch.tensor([1.0, 1.0, 0.0], dtype=overlay.dtype).view(3, 1)

    overlay[:, tp] = overlay[:, tp] * (1 - alpha) + green * alpha
    overlay[:, fp] = overlay[:, fp] * (1 - alpha) + red * alpha
    overlay[:, fn] = overlay[:, fn] * (1 - alpha) + yellow * alpha

    return overlay.clamp(0, 1)


def compute_sample_iou(pred_mask, gt_mask, eps=1e-6):
    pred = pred_mask.bool()
    gt = gt_mask.bool()

    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()

    if union == 0:
        return 1.0
    return (intersection / (union + eps)).item()


def compute_sample_metrics(pred_mask, gt_mask, eps=1e-6):
    pred = pred_mask.bool()
    gt = gt_mask.bool()

    tp = ((pred == 1) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()

    if (tp + fp + fn) == 0:
        iou = 1.0
        f1 = 1.0
    else:
        iou = tp / (tp + fp + fn + eps)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "f1": f1,
    }


def save_sample_metrics_csv(save_root, rows, filename="per_image_metrics.csv"):
    csv_path = os.path.join(save_root, filename)
    fieldnames = ["file_name", "tp", "tn", "fp", "fn", "iou", "f1"]

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def save_segmentation_mismatch_samples(
    save_root,
    images,
    gt_masks,
    pred_masks,
    file_names,
    batch_idx,
    iou_threshold=None,
):
    """
    segmentation mismatch 샘플 저장
    - overlay
    - original image
    - gt mask
    - pred mask

    iou_threshold:
        None  -> pred != gt 인 경우 전부 저장
        float -> sample_iou < threshold 인 경우만 저장
    """
    overlay_dir = os.path.join(save_root, "overlay")
    img_dir = os.path.join(save_root, "images")
    gt_dir = os.path.join(save_root, "gt_masks")
    pred_dir = os.path.join(save_root, "pred_masks")

    ensure_dir(overlay_dir)
    ensure_dir(img_dir)
    ensure_dir(gt_dir)
    ensure_dir(pred_dir)

    saved_count = 0

    for i in range(images.size(0)):
        sample_iou = compute_sample_iou(pred_masks[i], gt_masks[i])

        # if iou_threshold is None:
        #     if torch.equal(pred_masks[i], gt_masks[i]):
        #         continue
        # else:
        #     if sample_iou >= iou_threshold:
        #         continue

        file_name = file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}.png"
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        img = images[i].detach().cpu().float()
        gt_mask = gt_masks[i].detach().cpu().float().squeeze()       # [H,W]
        pred_mask = pred_masks[i].detach().cpu().float().squeeze()   # [H,W]

        overlay = make_overlay(img, gt_mask, pred_mask, alpha=0.45)

        save_image(overlay, os.path.join(overlay_dir, f"{base_name}_overlay.png"))
        save_image(img, os.path.join(img_dir, f"{base_name}_iou_{sample_iou:.3f}.png"))
        save_image(gt_mask.unsqueeze(0), os.path.join(gt_dir, f"{base_name}_gt.png"))
        save_image(pred_mask.unsqueeze(0), os.path.join(pred_dir, f"{base_name}_pred.png"))

        saved_count += 1

    return saved_count


def build_test_dataset(args):
    if args.dataset == "HRCUS_FAKE":
        return HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/test", split="test")
    elif args.dataset == "Fake-LoveDA":
        return Fake_LoveDA(root_dir="../dataset/Fake-LoveDA", split="test")
    elif args.dataset == "Fake-Vaihingen":
        return Fake_Vaihingen_LoveDA(root_dir="./dataset/Fake-Vaihingen", split="test")
    else:
        raise ValueError(f"지원하지 않는 dataset: {args.dataset}")


def test(model, test_loader, device, save_root):
    model.eval()

    test_tp, test_tn, test_fp, test_fn = 0.0, 0.0, 0.0, 0.0
    saved_seg_mismatch = 0
    per_image_rows = []

    seg_mismatch_root = os.path.join(save_root, "seg_mismatch_samples")
    ensure_dir(seg_mismatch_root)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device)
            gt_mask = batch["mask"].to(device)
            file_names = batch.get(
                "file_name",
                [f"batch{batch_idx}_{i}.png" for i in range(data.size(0))]
            )

            if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
                gt_mask = gt_mask.squeeze(1)

            gt_mask = gt_mask.long()

            output = model(data)
            seg_pred, cls_pred = output

            # seg_pred: [B,C,H,W]
            pred_mask = seg_pred.argmax(dim=1).long()  # [B,H,W]

            # per-image metrics
            for i in range(pred_mask.size(0)):
                file_name = file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}.png"
                metrics = compute_sample_metrics(pred_mask[i], gt_mask[i])
                metrics["file_name"] = file_name
                per_image_rows.append(metrics)

            # batch metrics
            tp, tn, fp, fn = comfusion_matrix(pred_mask, gt_mask)
            test_tp += tp
            test_tn += tn
            test_fp += fp
            test_fn += fn

            # mismatch save
            saved_seg_mismatch += save_segmentation_mismatch_samples(
                save_root=seg_mismatch_root,
                images=data,
                gt_masks=gt_mask,
                pred_masks=pred_mask,
                file_names=file_names,
                batch_idx=batch_idx,
                iou_threshold=None,   # 필요하면 0.5 같은 값 넣기
            )

    csv_path = save_sample_metrics_csv(save_root, per_image_rows)

    test_seg_mf1, test_seg_miou, test_seg_oa = compute_seg_metrics(
        test_tp, test_tn, test_fp, test_fn
    )

    print("\n[Test Result]")
    print(f"Seg mF1 : {test_seg_mf1:.4f}")
    print(f"Seg mIoU: {test_seg_miou:.4f}")
    print(f"Seg OA  : {test_seg_oa:.4f}")
    print(f"Saved segmentation mismatches: {saved_seg_mismatch}")
    print(f"Per-image metrics saved to: {csv_path}")
    print(f"Seg mismatch folder: {seg_mismatch_root}")


def main(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dataset = build_test_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = FLDCF(args).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"Loaded checkpoint from: {args.model_path}")

    save_root = os.path.join(args.save_dir, args.dataset, args.output_name)
    ensure_dir(save_root)

    test(model, test_loader, device, save_root)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="FLDCF test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="Fake-Vaihingen",
                        help="HRCUS_FAKE, Fake-LoveDA, Fake-Vaihingen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", type=str, default="fldcf_fakeV_test")
    parser.add_argument("--save_dir", type=str, default="test_output")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\KimSeowon\Desktop\kimseowon_Research\CounterPart_Model\train\fldcf_fakeV1_b8_lr0.0001\epoch_72_trainF1_0.9813_trainmIOU_0.9636.pth",
        help="best_checkpoint.pth or model .pth path"
    )

    args = parser.parse_args()
    main(args)