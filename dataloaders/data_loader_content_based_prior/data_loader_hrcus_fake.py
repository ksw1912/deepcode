import os
from torch.utils.data import Dataset
from torchvision.io import decode_image

# train/val/test folder structure

"""
How to call the HRCUS_FAKE dataset

from torch.utils.data import DataLoader

train_dataset = HRCUS_FAKE(root_dir="root", split="train")
val_dataset = HRCUS_FAKE(root_dir="root", split="val")
test_dataset = HRCUS_FAKE(root_dir="root", split="test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
"""


class HRCUS_FAKE(Dataset):
    def __init__(self, split, root_dir='./dataset/HRCUS_fakev16', transform=None,
                 target_transform=None):  # train/test/val call 'split' ex split = test
        self.root_dir = root_dir
        self.split = split

        self.img_dir = os.path.join(root_dir, split, "images")
        self.label_dir = os.path.join(root_dir, split, "labels")

        self.file_names = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]

        image_path = os.path.join(self.img_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        image = decode_image(image_path).float() / 255.0
        label = decode_image(label_path).float()

        # # If the label has 3 channels, only 1 channel is used.
        if label.shape[0] > 1:
            label = label[:1, :, :]

        # 0/1 Binarization
        label = (label > 0).float()

        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch

    train_dataset = HRCUS_FAKE(split="train")
    val_dataset = HRCUS_FAKE(split="val")
    test_dataset = HRCUS_FAKE(split="test")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(images.shape)
        break


"""
import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--dataset",  type=str,  default=HRCUS_FAKE, help="dataset setting(HRCUS_FAKE, )")




"""
