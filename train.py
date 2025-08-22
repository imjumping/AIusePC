# train.py - AIusePC 项目训练脚本
# 项目: https://github.com/imjumping/AIusePC
# 作者: imjumping (12岁)
# 用途: 训练 AI 学习鼠标行为

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# 防止路径问题
sys.path.append(os.path.dirname(__file__))

# =============== 重要提示 ===============
print("""
🚀 欢迎使用 AIusePC 训练系统！
📌 请确保：
   - 视频文件位于: dataset/video.mp4
   - 鼠标数据位于: dataset/mouse_data/data.txt
   - 使用 Linux + GNOME/KDE Plasma 桌面环境效果最佳

⚠️  本项目仅用于娱乐，切勿在主力电脑上运行！
❗ 如果 AI 行为异常，请立即终止程序！

开始加载数据集...
""")
# ======================================

# 尝试导入自定义模块（dataset_loader.py 和 model.py）
try:
    from dataset_loader import MouseDataset
    from model import MousePredictor
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 dataset_loader.py 和 model.py 与 train.py 在同一目录下！")
    sys.exit(1)

# ================================
# 超参数设置
# ================================
BATCH_SIZE = 32
EPOCHS = 15                  # 增加训练轮数以更好拟合
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 2
PIN_MEMORY = True if torch.cuda.is_available() else False

# 数据路径（严格按照 README 要求）
VIDEO_PATH = "dataset/video.mp4"
LABEL_PATH = "dataset/mouse_data/data.txt"

# 检查文件是否存在
if not os.path.exists(VIDEO_PATH):
    print(f"❌ 错误: 未找到视频文件 '{VIDEO_PATH}'")
    print("请将录制的屏幕视频保存为 dataset/video.mp4")
    sys.exit(1)

if not os.path.exists(LABEL_PATH):
    print(f"❌ 错误: 未找到鼠标标注文件 '{LABEL_PATH}'")
    print("请将鼠标数据保存为 dataset/mouse_data/data.txt")
    sys.exit(1)

# 最大训练帧数（None 表示全部训练）
MAX_FRAMES = None  # 可设为 1000 用于测试

# 模型保存路径（严格按照 README）
MODEL_DIR = "model"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.bin")  # 必须是 model.bin

# 创建 model 目录
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# 数据集加载
# ================================
print("🔍 正在加载数据集...")
try:
    dataset = MouseDataset(
        label_path=LABEL_PATH,
        max_frames=MAX_FRAMES
    )
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    print("请检查 video.mp4 和 data.txt 格式是否正确。")
    sys.exit(1)

if len(dataset) == 0:
    print("❌ 数据集为空！请检查 data.txt 是否有有效数据。")
    sys.exit(1)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True
)

print(f"✅ 数据集加载成功！共 {len(dataset)} 帧数据，每个 epoch {len(dataloader)} 个 batch。")

# ================================
# 模型定义
# ================================
print(f"🧠 正在构建模型... 使用设备: {DEVICE}")
try:
    model = MousePredictor().to(DEVICE)
except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    sys.exit(1)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5轮降学习率

# 损失函数
pos_criterion = nn.MSELoss()        # 坐标回归
click_criterion = nn.BCELoss()      # 是否点击（二分类）

# ================================
# 开始训练
# ================================
print("\n" + "="*50)
print("🚀 开始训练 AI 模型...")
print("="*50)

model.train()
best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0.0
    pos_loss_acc = 0.0
    click_loss_acc = 0.0
    batch_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False)

    for images, coords, clicks in progress_bar:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        coords = coords.to(DEVICE, non_blocking=PIN_MEMORY)
        clicks = clicks.float().to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad()
        pred_coords, pred_clicks = model(images)

        loss_pos = pos_criterion(pred_coords, coords)
        loss_click = click_criterion(pred_clicks, clicks)
        total_loss_step = loss_pos + loss_click

        total_loss_step.backward()
        optimizer.step()

        total_loss += total_loss_step.item()
        pos_loss_acc += loss_pos.item()
        click_loss_acc += loss_click.item()
        batch_count += 1

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{total_loss_step.item():.4f}',
            'pos': f'{loss_pos.item():.4f}',
            'click': f'{loss_click.item():.4f}'
        })

    avg_loss = total_loss / batch_count
    avg_pos_loss = pos_loss_acc / batch_count
    avg_click_loss = click_loss_acc / batch_count

    print(f"✅ Epoch {epoch+1}: "
          f"总损失={avg_loss:.4f} | "
          f"坐标损失={avg_pos_loss:.4f} | "
          f"点击损失={avg_click_loss:.4f}")

    # 学习率调度
    scheduler.step()

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 最佳模型已保存至: {MODEL_SAVE_PATH} (loss: {best_loss:.4f})")

# ================================
# 训练完成
# ================================
print("\n" + "="*60)
print("🎉 训练完成！")
print(f"📍 模型已保存至: {MODEL_SAVE_PATH}")
print(f"📊 最佳损失: {best_loss:.4f}")
print("="*60)
print("""
✨ 接下来：
   1. 运行 main.py 让 AI 控制你的电脑！
   2. ⚠️ 危险行为？立刻按 ESC 两次退出！
   3. ❗ 本项目仅供娱乐，切勿日常使用！

😁 祝你好运！
GitHub: github.com/imjumping
""")