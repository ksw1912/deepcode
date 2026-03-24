import os
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch
import random
# train/test folder structure

"""
How to call the HRCUS_FAKE dataset

from torch.utils.data import DataLoader

train_dataset = Fake_Vaihingen(root_dir="root", split="train")
val_dataset = HRCUS_FAKE(root_dir="root", split="val")
test_dataset = HRCUS_FAKE(root_dir="root", split="test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
"""


class Fake_Vaihingen_Prior(Dataset):
    def __init__(self, root_dir='./dataset/Fake-Vaihingen',
                 split="train"):  # train/test/val call 'split' ex split = test
        self.root_dir = root_dir
        self.split = split
        self.gt_dir = os.path.join(self.root_dir, self.split, "gt")  # Authentic img_dir
        self.lama_dir = os.path.join(self.root_dir, self.split, "lama")  # fake img1_dir
        self.repaint_dir = os.path.join(self.root_dir, self.split, "repaint")  # fake img2_dir


        self.lama_files = sorted(os.listdir(self.lama_dir))
        self.repaint_files = sorted(os.listdir(self.repaint_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))
        self.real_fake = [ ]
        """
        EX:
        self.real_fake = [
            {
                'real': 'gt/top_001.png',
                'fake': {
                    'lama': 'train/lama/top_001.png',
                    'repaint': 'train/repaint/top_001.png'
                }
            },
            ...
        ]
        """

        # 파일명 기준 매핑
        gt_map = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in self.gt_files
        }
        lama_map = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in self.lama_files
        }
        repaint_map = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in self.repaint_files
        }

        for name in gt_map:
            entry = {
                'real': gt_map[name],
                'fake': {}
            }

            if name in lama_map:
                entry['fake']['lama'] = lama_map[name]

            if name in repaint_map:
                entry['fake']['repaint'] = repaint_map[name]

            # fake가 하나라도 있어야 의미 있음
            if len(entry['fake']) > 0:
                self.real_fake.append(entry)
        print(len(self.real_fake))

    def __len__(self):
        return len(self.real_fake)

    def __getitem__(self, idx):
        pair = self.real_fake[idx]

        real_path = pair['real']
        fake_dict = pair['fake']

        fake_path = random.choice(list(fake_dict.values()))

        real  = decode_image(real_path).float() / 255.0
        fake  = decode_image(fake_path).float() / 255.0

        # If the label has 3 channels, only 1 channel is used.

        return real, fake


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split

    train_dataset = Fake_Vaihingen_Prior(root_dir='../../dataset/Fake-Vaihingen', split="train")
    test_dataset = Fake_Vaihingen_Prior('../../dataset/Fake-Vaihingen',split="test")

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset_split, val_dataset_split = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset_split, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=8, shuffle=False)


    print("train:", len(train_dataset_split))
    print("val:", len(val_dataset_split))
    print("test:", len(test_dataset))
    # for a in train_loader:
    #     print(a['image'].shape, a['image'].shape)
    #     # print(images.shape)
    #     break
