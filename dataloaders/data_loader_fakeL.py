import os
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch

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


class Fake_LoveDA(Dataset):
    def __init__(self, root_dir='../dataset/Fake-LoveDA',
                 split="train"):  # train/test/val call 'split' ex split = test
        self.root_dir = root_dir
        self.split = split
        self.gt_dir = os.path.join(self.root_dir, self.split, "gt")  # Authentic img_dir
        self.gt_mask_path = os.path.join(self.root_dir, "gt_mask", "gt_mask.png")  # Aututhentic gt img

        self.lama_dir = os.path.join(self.root_dir, self.split, "lama")  # fake img1_dir
        self.repaint_dir = os.path.join(self.root_dir, self.split, "repaint")  # fake img2_dir
        self.inpainted_mask_dir = os.path.join(self.root_dir, self.split, "inpainted_mask")  # fake_gt img_dir


        #train/test folder의 image file name lists
        self.lama_files = sorted(os.listdir(self.lama_dir))
        self.repaint_files = sorted(os.listdir(self.repaint_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))

        print("lama_files:", len(self.lama_files),"repaint_files",len(self.repaint_files),"gt_files",len(self.gt_files))

        self.fake_files = []
        self.real_files = []
        # self.samples = None

        for f in self.lama_files:
            mask_path = os.path.join(self.inpainted_mask_dir, f)
            if os.path.exists(mask_path):
                self.fake_files.append({
                    "file_name": f,
                    "img_dir": self.lama_dir,
                    "mask_path": mask_path,
                    "source": "lama",
                    "is_fake": 1
                })

        for f in self.repaint_files:
            mask_path = os.path.join(self.inpainted_mask_dir, f)
            if os.path.exists(mask_path):
                self.fake_files.append({
                    "file_name": f,
                    "img_dir": self.repaint_dir,
                    "mask_path": mask_path,
                    "source": "repaint",
                    "is_fake": 1
                })

        def get_idx(name):
            stem = os.path.splitext(name)[0]
            return int(stem.split("_")[-1])

        self.fake_files = sorted(self.fake_files, key=lambda x: get_idx(x["file_name"]))

        gt_files = sorted([
            f for f in os.listdir(self.gt_dir)
            if os.path.isfile(os.path.join(self.gt_dir, f))
        ])
        for f in gt_files:
            self.real_files.append({
                "file_name": f,
                "img_dir": self.gt_dir,
                "mask_path": self.gt_mask_path,
                "source": "real",
                "is_fake": 0
            })

        self.samples = self.fake_files + self.real_files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = os.path.join(sample["img_dir"], sample["file_name"])

        if sample["is_fake"] == 1:
            mask_path = sample["mask_path"]


        else:
            mask_path = self.gt_mask_path  # real image는 all-zero mask

        image = decode_image(img_path).float() / 255.0
        mask = decode_image(mask_path).float()

        # If the label has 3 channels, only 1 channel is used.
        if mask.shape[0] > 1:
            mask = mask[0]
        else:
            mask = mask.squeeze(0)

        # 0/1 Binarization
        # mask = (mask > 0).float()
        mask = (mask > 0).long()

        return {
            "image": image,
            "mask": mask,
            "is_fake": torch.tensor(sample["is_fake"], dtype=torch.long),  # fake=1, real=0
            "file_name": sample["file_name"]
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader,  random_split

    train_dataset = Fake_LoveDA(root_dir='../dataset/Fake-LoveDA', split="train")
    test_dataset = Fake_LoveDA('../dataset/Fake-LoveDA',split="test")

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
