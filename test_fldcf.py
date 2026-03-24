import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataloaders.data_loader_hrcus_fake import HRCUS_FAKE
from dataloaders.data_loader_fakeV import Fake_Vaihingen
from dataloaders.data_loader_fakeL import Fake_LoveDA
from models.fldcf_dir.fldcf import FLDCF
from train.seed_setting import set_seed
from utills.metrics import compute_seg_metrics, comfusion_matrix


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_mismatch_samples(
    save_root,
    images,
    gt_masks,
    pred_masks,
    pred_cls,
    gt_cls,
    file_names,
    batch_idx
):
    """
    classification 예측이 틀린 샘플만 저장
    저장 파일:
      - 원본 이미지
      - gt mask
      - pred mask
    """
    mismatch_idx = (pred_cls != gt_cls).nonzero(as_tuple=False).squeeze(1)

    if mismatch_idx.numel() == 0:
        return 0

    img_dir = os.path.join(save_root, "images")
    gt_dir = os.path.join(save_root, "gt_masks")
    pred_dir = os.path.join(save_root, "pred_masks")

    ensure_dir(img_dir)
    ensure_dir(gt_dir)
    ensure_dir(pred_dir)

    saved_count = 0

    for i in mismatch_idx.tolist():
        file_name = file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}.png"
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        img = images[i].detach().cpu()
        gt_mask = gt_masks[i].detach().cpu().float().unsqueeze(0)   # [1,H,W]
        pred_mask = pred_masks[i].detach().cpu().float().unsqueeze(0)

        # mask는 보기 쉽게 0/1 -> 0/255 느낌으로 저장되도록 그대로 저장
        save_image(img, os.path.join(img_dir, f"{base_name}.png"))
        save_image(gt_mask, os.path.join(gt_dir, f"{base_name}_gt.png"))
        save_image(pred_mask, os.path.join(pred_dir, f"{base_name}_pred.png"))

        saved_count += 1

    return saved_count

def compute_sample_iou(pred_mask, gt_mask, eps=1e-6):
    """
    pred_mask: [H, W] (0/1)
    gt_mask:   [H, W] (0/1)
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()

    if union == 0:
        return 1.0  # 둘 다 전부 background면 완전 일치로 간주
    return (intersection / (union + eps)).item()

# def save_segmentation_mismatch_samples(
#     save_root,
#     images,
#     gt_masks,
#     pred_masks,
#     file_names,
#     batch_idx,
#     iou_threshold=0.5
# ):
#     """
#     IoU가 threshold보다 낮은 샘플 저장
#     """
#     img_dir = os.path.join(save_root, "images")
#     gt_dir = os.path.join(save_root, "gt_masks")
#     pred_dir = os.path.join(save_root, "pred_masks")
#
#     ensure_dir(img_dir)
#     ensure_dir(gt_dir)
#     ensure_dir(pred_dir)
#
#     saved_count = 0
#
#     for i in range(images.size(0)):
#         sample_iou = compute_sample_iou(pred_masks[i], gt_masks[i])
#
#         if sample_iou >= iou_threshold:
#             continue
#
#         file_name = file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}.png"
#         base_name = os.path.splitext(os.path.basename(file_name))[0]
#
#         img = images[i].detach().cpu()
#         gt_mask = gt_masks[i].detach().cpu().float().unsqueeze(0)
#         pred_mask = pred_masks[i].detach().cpu().float().unsqueeze(0)
#
#         save_image(img, os.path.join(img_dir, f"{base_name}_iou_{sample_iou:.3f}.png"))
#         save_image(gt_mask, os.path.join(gt_dir, f"{base_name}_gt.png"))
#         save_image(pred_mask, os.path.join(pred_dir, f"{base_name}_pred.png"))
#
#         saved_count += 1
#
#     return saved_count

def save_segmentation_mismatch_samples(
    save_root,
    images,
    gt_masks,
    pred_masks,
    file_names,
    batch_idx
):
    img_dir = os.path.join(save_root, "images")
    gt_dir = os.path.join(save_root, "gt_masks")
    pred_dir = os.path.join(save_root, "pred_masks")

    ensure_dir(img_dir)
    ensure_dir(gt_dir)
    ensure_dir(pred_dir)

    saved_count = 0

    for i in range(images.size(0)):
        if torch.equal(pred_masks[i], gt_masks[i]):
            continue

        file_name = file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}.png"
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        img = images[i].detach().cpu()
        gt_mask = gt_masks[i].detach().cpu().float().unsqueeze(0)
        pred_mask = pred_masks[i].detach().cpu().float().unsqueeze(0)

        save_image(img, os.path.join(img_dir, f"{base_name}.png"))
        save_image(gt_mask, os.path.join(gt_dir, f"{base_name}_gt.png"))
        save_image(pred_mask, os.path.join(pred_dir, f"{base_name}_pred.png"))

        saved_count += 1

    return saved_count

def build_test_dataset(args):
    if args.dataset == "HRCUS_FAKE":
        test_dataset = HRCUS_FAKE(root_dir="./dataset/HRCUS_fakev16/test", split="test")

    elif args.dataset == "Fake-LoveDA":
        test_dataset = Fake_LoveDA(root_dir="../dataset/Fake-LoveDA", split="test")

    elif args.dataset == "Fake-Vaihingen":
        test_dataset = Fake_Vaihingen(root_dir="./dataset/Fake-Vaihingen", split="test")

    else:
        raise ValueError(f"지원하지 않는 dataset: {args.dataset}")

    return test_dataset


def test(model, test_loader, device, save_root):
    model.eval()

    total_samples = 0
    total_correct = 0

    test_tp, test_tn, test_fp, test_fn = 0.0, 0.0, 0.0, 0.0
    saved_mismatch = 0

    mismatch_root = os.path.join(save_root, "mismatch_samples")
    ensure_dir(mismatch_root)

    saved_seg_mismatch = 0
    mismatch_root = os.path.join(save_root, "saved_seg_mismatch")
    ensure_dir(mismatch_root)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device)
            gt_mask = batch["mask"].to(device)
            cls_label = batch["is_fake"].to(device)
            file_names = batch.get("file_name", [f"batch{batch_idx}_{i}.png" for i in range(data.size(0))])

            if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
                gt_mask = gt_mask.squeeze(1)

            gt_mask = gt_mask.long()

            output = model(data)
            seg_pred, cls_pred = output

            # segmentation
            pred_mask = seg_pred.argmax(dim=1).long()  # [B,H,W]

            tp, tn, fp, fn = comfusion_matrix(pred_mask, gt_mask)
            test_tp += tp
            test_tn += tn
            test_fp += fp
            test_fn += fn

            # classification
            pred_cls = cls_pred.argmax(dim=1)  # [B]
            total_correct += (pred_cls == cls_label).sum().item()
            total_samples += cls_label.size(0)

            # mismatch 저장
            saved_mismatch += save_mismatch_samples(
                save_root=mismatch_root,
                images=data,
                gt_masks=gt_mask,
                pred_masks=pred_mask,
                pred_cls=pred_cls,
                gt_cls=cls_label,
                file_names=file_names,
                batch_idx=batch_idx
            )

            saved_seg_mismatch += save_segmentation_mismatch_samples(
                save_root=os.path.join(save_root, "seg_mismatch_samples"),
                images=data,
                gt_masks=gt_mask,
                pred_masks=pred_mask,
                file_names=file_names,
                batch_idx=batch_idx
                # ,iou_threshold=0.5
            )

    test_seg_mf1, test_seg_miou, test_seg_oa = compute_seg_metrics(
        test_tp, test_tn, test_fp, test_fn
    )
    test_cls_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    print("\n[Test Result]")
    print(f"Cls Acc : {test_cls_acc:.2f}%")
    print(f"Seg mF1 : {test_seg_mf1:.4f}")
    print(f"Seg mIoU: {test_seg_miou:.4f}")
    print(f"Seg OA  : {test_seg_oa:.4f}")
    print(f"Saved mismatched samples: {saved_mismatch}")
    print(f"Mismatch folder: {mismatch_root}")


def main(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dataset = build_test_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = FLDCF(args)
    model.to(device)

    ckpt = torch.load(args.model_path, map_location=device)

    # state_dict만 저장한 경우 / checkpoint dict 저장한 경우 둘 다 대응
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
    parser.add_argument("--model_path", type=str, default="./trained_output/fldcf_fakeV_b8_lr0.0001/epoch_185_trainF1_0.9942498207092285_trainmIOU_0.9885954856872559.pth",
                        help="best_checkpoint.pth or model .pth path")

    args = parser.parse_args()
    main(args)
