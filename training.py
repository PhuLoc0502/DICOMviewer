import os
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    EnsureChannelFirst, ScaleIntensity, Resize, Compose, ToTensor
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.optim import Adam, AdamW

# ĐƯỜNG DẪN TỚI DỮ LIỆU
folder_path = "E:\\Unet_MonAi_CXR2D\\train02"
images_path = os.path.join(folder_path, "images")
masks_path = os.path.join(folder_path, "masks")

# Lấy danh sách file và sắp xếp
image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path)])
mask_files = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path)])

# GHÉP CẶP ẢNH VÀ MASK
paired_files = list(zip(image_files, mask_files))

# Chia train/val (80% train, 20% val)
random.shuffle(paired_files)
split_idx = int(len(paired_files) * 0.8)
train_files = paired_files[:split_idx]
val_files = paired_files[split_idx:]

# HÀM ĐỌC ẢNH & CHUẨN BỊ TRANSFORMS
def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Resize ảnh về kích thước cố định
    img = img.astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
    img = np.expand_dims(img, axis=0)  # Thêm chiều kênh (1, H, W)
    return img

# Dataset MONAI
class XRaySegmentationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = load_image(img_path)
        mask = load_image(mask_path)
        return {"image": torch.tensor(img, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32)}

# Tạo Dataset & DataLoader
train_ds = XRaySegmentationDataset(train_files)
val_ds = XRaySegmentationDataset(val_files)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# THIẾT LẬP MÔ HÌNH UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# LOSS, OPTIMIZER, METRICS
loss_function = DiceLoss(sigmoid=True)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# HUẤN LUYỆN MÔ HÌNH
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        inputs, masks = batch["image"].to(device), batch["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# LƯU MÔ HÌNH
torch.save(model.state_dict(), "unet_xray_segmentation03.pth")
print("Training completed and model saved!")
