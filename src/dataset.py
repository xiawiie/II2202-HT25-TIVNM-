import os
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
from src.config import Config


class RSNADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train', full_df_path=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        # 仅在评估模式且提供了完整CSV时加载 Bbox
        if mode == 'eval' and full_df_path:
            self.bbox_df = pd.read_csv(full_df_path)
        else:
            self.bbox_df = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['patientId']
        target = int(row['Target'])

        dcm_path = os.path.join(self.img_dir, f"{pid}.dcm")
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
        except Exception as e:
            img = np.zeros((1024, 1024))

        # 归一化处理
        if np.max(img) > 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)

        # 记录原始尺寸
        orig_h, orig_w = img.shape

        # 转 RGB
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)

        img_pil = Image.fromarray(img)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # 评估模式返回 Mask
        if self.mode == 'eval':
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if target == 1 and self.bbox_df is not None:
                boxes = self.bbox_df[self.bbox_df['patientId'] == pid]
                for _, box in boxes.iterrows():
                    if not pd.isna(box['x']):
                        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
                        cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)

            mask = cv2.resize(mask, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            return img_tensor, target, torch.from_numpy(mask).long(), pid

        return img_tensor, torch.tensor(target, dtype=torch.float32)