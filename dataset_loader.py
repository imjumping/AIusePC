# dataset_loader.py - 极速版，直接读图片

import torch
import cv2
import numpy as np
import os

class MouseDataset:
    def __init__(self, label_path="dataset/mouse_data/data.txt", max_frames=None):
        self.data = []

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    frame_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    buttons = parts[3:]

                    img_path = f"dataset/frames/{frame_id:06d}.jpg"
                    if not os.path.exists(img_path):
                        print(f"⚠️ 图片不存在: {img_path}")
                        continue

                    # 假设屏幕 1920x1080
                    norm_x = x / 1920
                    norm_y = y / 1080
                    is_pressed = 0 if "nomouse" in buttons else 1

                    self.data.append({
                        'img_path': img_path,
                        'coords': np.array([norm_x, norm_y], dtype=np.float32),
                        'pressed': is_pressed
                    })
                except Exception as e:
                    print(f"跳过无效行: {line}, 错误: {e}")

        if max_frames:
            self.data = self.data[:max_frames]

        print(f"✅ 加载了 {len(self.data)} 帧数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item['img_path'])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {item['img_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW

        return image, torch.tensor(item['coords']), torch.tensor(item['pressed'])