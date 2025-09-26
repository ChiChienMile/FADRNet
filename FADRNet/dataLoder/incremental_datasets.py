import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# from dataload_t1c import default_loader, default_loader_test
# from dataload_t1c_freq import default_loader_freq, default_loader_test_freq

from dataLoder.dataload_t1c import default_loader, default_loader_test
from dataLoder.dataload_t1c_freq import default_loader_freq, default_loader_test_freq
# =========================================================
# Dataset loaders for three incremental prediction tasks:
#   1p19q / LHG / IDH
# Provide both standard loaders and frequency-masked loaders
# =========================================================

# ===============================
# Standard Training Dataset
# ===============================
class Dataset_Train(Dataset):
    def __init__(self, transform, task, read_type='train'):
        """
        Args:
            transform: preprocessing pipeline (e.g., monai.transforms.ToTensor())
            task (str): task name in {"1p19q", "LHG", "IDH"} (free-form string accepted)
            read_type (str): split name, typically {"train","val","test"}
        """
        self.read_type = read_type
        self.transform = transform

        # Load metadata from pre-saved pickle: expects keys {read_name, path_dir_all, labels}
        save_path = f"./dataLoder/{task}_{read_type}.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.read_name = data["read_name"]       # sample IDs
        self.path_dir_all = data["path_dir_all"] # image paths
        self.labels = data["labels"]             # labels

    def __len__(self):
        return len(self.read_name)

    def match_tensor(self, imgs_img):
        """Apply transform and add a leading dimension (kept for compatibility)."""
        imgs_img = self.transform(imgs_img)
        pos_img = imgs_img.unsqueeze(0)
        return pos_img

    def __getitem__(self, idx):
        imgs = default_loader(self.path_dir_all[idx])    # load image(s)
        cls_label = int(self.labels[idx])                # load label
        cls_label = self.transform(cls_label)            # tensorize label
        pos_img = self.match_tensor(imgs)                # preprocess image
        return pos_img.float(), cls_label.long()


# ===============================
# Standard Test Dataset
# ===============================
class Dataset_Test(Dataset):
    def __init__(self, transform, task, read_type='test', nameflag=False):
        """
        Args:
            transform: preprocessing pipeline
            task (str): task name
            read_type (str): split name (default: "test")
            nameflag (bool): if True, also return sample ID and path
        """
        self.read_type = read_type
        self.transform = transform
        self.nameflag = nameflag

        # Load metadata
        save_path = f"./dataLoder/{task}_{read_type}.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.read_name = data["read_name"]
        self.path_dir_all = data["path_dir_all"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.path_dir_all)

    def match_tensor(self, imgs_img):
        """Apply transform and add a leading dimension (kept for compatibility)."""
        imgs_img = self.transform(imgs_img)
        pos_img = imgs_img.unsqueeze(0)
        return pos_img

    def __getitem__(self, idx):
        # The default test loader can return (image, mask); we keep mask unused to preserve behavior.
        imgs = default_loader_test(self.path_dir_all[idx])
        cls_label = int(self.labels[idx])
        cls_label = self.transform(cls_label)
        pos_img = self.match_tensor(imgs)

        if self.nameflag:
            return pos_img.float(), cls_label.long(), self.read_name[idx], self.path_dir_all[idx]
        else:
            return pos_img.float(), cls_label.long()


# =========================================================
# Frequency-Masked Datasets
# diameters_list example: [70, 110, 150, 999]
# - Each diameter defines a spherical mask in k-space
# - 999 means "no filtering" (use the raw image)
# NOTE: We keep the original nested-loop mask generation for 1:1 behavior.
# =========================================================

class Dataset_Train_frequency(Dataset):
    def __init__(self, transform, diameters_list, task, read_type='train'):
        """
        Args:
            transform: preprocessing pipeline
            diameters_list (List[int]): list of frequency cutoff radii (999 => no filtering)
            task (str): task name
            read_type (str): split name
        """
        self.read_type = read_type
        self.transform = transform
        self.diameters_list = diameters_list

        # Load metadata
        save_path = f"./dataLoder/{task}_{read_type}.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.read_name = data["read_name"]
        self.path_dir_all = data["path_dir_all"]
        self.labels = data["labels"]

        # Build masks for all diameters
        mask_list = []
        shape = [160, 192, 160]
        center = np.array([dim // 2 for dim in shape])
        for d in diameters_list:
            if d == 999:
                mask = np.ones(shape, dtype=bool)
            else:
                mask = np.zeros(shape, dtype=bool)
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        for z in range(shape[2]):
                            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                            if distance <= (d / 2):
                                mask[x, y, z] = True
            mask_list.append(mask)
        self.freqmask = mask_list

    def __len__(self):
        return len(self.read_name)

    def match_tensor(self, imgs):
        """Convert a list of frequency-split images to float tensors with leading channel dim."""
        img_all = []
        for img in imgs:
            img = self.transform(img)
            img = img.unsqueeze(0)  # preserve original behavior
            img_all.append(img.float())
        return img_all

    def __getitem__(self, idx):
        imgs = default_loader_freq(self.path_dir_all[idx], self.diameters_list, self.freqmask)
        cls_label = int(self.labels[idx])
        cls_label = self.transform(cls_label)
        img_all = self.match_tensor(imgs)
        return img_all, cls_label.long()


class Dataset_Test_frequency(Dataset):
    def __init__(self, transform, diameters_list, task, read_type='test', show_freq_mask=False, nameflag=False):
        """
        Args:
            transform: preprocessing pipeline
            diameters_list (List[int]): list of frequency cutoff radii (999 => no filtering)
            task (str): task name
            read_type (str): split name (default: "test")
            show_freq_mask (bool): if True, visualize one mask (debug)
            nameflag (bool): if True, also return sample ID and path
        """
        self.read_type = read_type
        self.transform = transform
        self.nameflag = nameflag
        self.diameters_list = diameters_list

        # Load metadata
        save_path = f"./dataLoder/{task}_{read_type}.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.read_name = data["read_name"]
        self.path_dir_all = data["path_dir_all"]
        self.labels = data["labels"]

        # Build masks for all diameters
        mask_list = []
        shape = [160, 192, 160]
        center = np.array([dim // 2 for dim in shape])
        for d in diameters_list:
            if d == 999:
                mask = np.ones(shape, dtype=bool)
            else:
                mask = np.zeros(shape, dtype=bool)
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        for z in range(shape[2]):
                            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                            if distance <= (d / 2):
                                mask[x, y, z] = True
            mask_list.append(mask)
            # Optional visualization of the first mask
            if show_freq_mask:
                import matplotlib.pyplot as plt
                mid_x, mid_y, mid_z = [dim // 2 for dim in shape]
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(mask[:, :, mid_z], cmap='gray')
                plt.title(f"Axial (z={mid_z})")
                plt.subplot(1, 3, 2)
                plt.imshow(mask[mid_x, :, :], cmap='gray')
                plt.title(f"Sagittal (x={mid_x})")
                plt.subplot(1, 3, 3)
                plt.imshow(mask[:, mid_y, :], cmap='gray')
                plt.title(f"Coronal (y={mid_y})")
                plt.tight_layout()
                plt.show()
        self.freqmask = mask_list

    def __len__(self):
        return len(self.path_dir_all)

    def match_tensor(self, imgs):
        """Convert a list of frequency-split images to float tensors with leading channel dim."""
        img_all = []
        for img in imgs:
            img = self.transform(img)
            img = img.unsqueeze(0)  # preserve original behavior
            img_all.append(img.float())
        return img_all

    def __getitem__(self, idx):
        imgs = default_loader_test_freq(self.path_dir_all[idx], self.diameters_list, self.freqmask)
        cls_label = int(self.labels[idx])
        cls_label = self.transform(cls_label)
        img_all = self.match_tensor(imgs)

        if self.nameflag:
            return img_all, cls_label.long(), self.read_name[idx], self.path_dir_all[idx]
        else:
            return img_all, cls_label.long()


# ===============================
# Minimal usage examples
# ===============================
if __name__ == '__main__':
    import monai

    # Compose a very simple transform that matches your existing pipeline
    val_transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),  # keep consistent with original behavior
    ])

    # -------- Standard Test Dataset example --------
    # Make sure the file "./_IDH_test.pkl" exists and loaders can read paths inside it.
    std_test_dataset = Dataset_Test(transform=val_transforms, task='IDH', read_type='test', nameflag=True)
    std_test_loader = DataLoader(
        std_test_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=False
    )
    print(f"[Standard Test] Number of examples: {len(std_test_dataset)}")
    for i, batch in enumerate(std_test_loader):
        imgs, labels, names, paths = batch
        print(f"[Standard Test] Batch {i}: imgs.shape={imgs.shape}, labels.shape={labels.shape}")
        break

    # -------- Frequency Test Dataset example (this is the requested test case) --------
    diameters_list = [70, 110, 150, 999]  # 999 => no filtering branch
    # freq_test_dataset = Dataset_Train_frequency(
    #     transform=val_transforms,
    #     diameters_list=diameters_list,
    #     task='IDH',
    #     read_type='test'
    # )
    freq_test_dataset = Dataset_Test_frequency(
        transform=val_transforms,
        diameters_list=diameters_list,
        task='IDH',
        read_type='test',
        show_freq_mask=True,   # set True to visualize the first mask
        nameflag=True
    )
    freq_test_loader = DataLoader(
        freq_test_dataset,
        batch_size=1,
        shuffle=False,          # typically deterministic for evaluation
        drop_last=False,
        num_workers=1,
        pin_memory=False
    )
    print(f"[Freq Test] Number of examples: {len(freq_test_dataset)}")
    for i, batch in enumerate(freq_test_loader):
        # freq branch returns a list of image tensors per sample
        img_list, labels, names, paths = batch
        print(f"[Freq Test] Sample {i}:")
        for bi, bimg in enumerate(img_list):
            # bimg has shape: [B, C, D, H, W] or similar depending on your loader/transform
            print(f"  - branch {bi} (diameter={diameters_list[bi]}): shape={tuple(bimg.shape)}")
        print(f"  labels.shape={labels.shape}, name[0]={names[0]}, path[0]={paths[0]}")
        break