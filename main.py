# main.py - AIusePC 主程序
# 让训练好的 AI 控制你的鼠标！
# 安全第一：双击 ESC 退出
# GitHub: github.com/imjumping

import torch
import cv2
import numpy as np
import pyautogui
import time
from pynput import keyboard, mouse
import os

# ================================
# 配置
# ================================
MODEL_PATH = "model/model.bin"
SCREEN_W, SCREEN_H = 2160, 1440  # 修改为你的分辨率
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（必须先定义模型结构）
from model import MousePredictor
model = MousePredictor().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型加载成功: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit(1)

# 实时截图工具（用 mss 更快，这里用 pyautogui 简单兼容）
def capture_screen():
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

# 安全退出机制
exit_count = 0
def on_press(key):
    global exit_count
    try:
        if key == keyboard.Key.esc:
            exit_count += 1
            if exit_count >= 2:
                print("\n🛑 双击 ESC 检测到，安全退出！")
                os._exit(0)
        else:
            exit_count = 0
    except:
        pass

# 启动键盘监听
listener = keyboard.Listener(on_press=on_press)
listener.start()
print("🎮 AI 已启动！")
print("💡 操作说明：")
print("   - AI 将自动控制鼠标")
print("   - ⚠️ 如果行为异常，立刻连按两次 ESC 退出！")
print("   - 本程序仅供娱乐，切勿日常使用！\n")
print("程序将在五秒后开始启动！")
time.sleep(5)
# ================================
# 主循环
# ================================
try:
    with torch.no_grad():
        while True:
            # 1. 截图
            frame = capture_screen()
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(DEVICE)

            # 2. 预测
            pred_pos, pred_click = model(frame_tensor)
            x_norm, y_norm = pred_pos[0].cpu().numpy()
            is_click = pred_click[0].item() > 0.5  # 阈值 0.5

            # 3. 转换为屏幕坐标
            x = int(x_norm * SCREEN_W)
            y = int(y_norm * SCREEN_H)

            # 4. 移动鼠标
            pyautogui.moveTo(x, y)

            # 5. 是否点击
            if is_click:
                pyautogui.click()

            # 6. 控制帧率（约 10 FPS）
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 程序被用户中断。")
except Exception as e:
    print(f"❌ 运行时错误: {e}")

finally:
    listener.stop()