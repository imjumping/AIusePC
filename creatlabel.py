# recorder.py - 高性能屏幕+鼠标记录器
# 优化点：mss + 直接保存为图片 + 高效标签

import mss
import mss.tools
import time as tm
import sys
from pynput import mouse
import os
import numpy as np

# ================================
# 配置参数
# ================================
FPS = 30
FRAME_DURATION = 1.0 / FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # 修改为你的屏幕分辨率，或用 pg.size()

# 自动获取屏幕大小（可选）
# import pyautogui
# SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# 数据保存路径
FRAME_DIR = "./dataset/frames"
LABEL_PATH = "./dataset/mouse_data/data.txt"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABEL_PATH), exist_ok=True)

# 鼠标状态
currently_pressed_buttons = set()

def on_click(x, y, button, pressed):
    global currently_pressed_buttons
    if pressed:
        currently_pressed_buttons.add(str(button))
    else:
        btn_str = str(button)
        if btn_str in currently_pressed_buttons:
            currently_pressed_buttons.discard(btn_str)

# 启动鼠标监听
listener = mouse.Listener(on_click=on_click)
listener.start()

# 打开标签文件
with open(LABEL_PATH, "w") as f:
    print("✅ 程序已启动，开始录制... 按 Ctrl+C 退出")

    # 使用 mss 进行高效截图
    with mss.mss() as sct:
        # 定义全屏区域
        monitor = {"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
        frame = 0

        try:
            while True:
                start_time = tm.time()

                # 📸 高性能截图
                img = sct.grab(monitor)
                # 保存为 JPG（质量可调，平衡大小与画质）
                output_path = f"{FRAME_DIR}/{frame:06d}.jpg"
                mss.tools.to_png(img.rgb, img.size, output=output_path)
                # 或保存为 .npy（更快但占空间）: np.save(output_path, np.array(img))

                # 🖱️ 获取鼠标位置
                # 注意：mss 坐标与 pyautogui 一致
                x, y = img.left + img.width // 2, img.top + img.height // 2  # 实际应获取全局坐标
                # 改为用 pyautogui 获取鼠标位置（更准）
                import pyautogui
                x, y = pyautogui.position()

                # 🔘 按键状态
                if currently_pressed_buttons:
                    pressed_str = ", ".join(currently_pressed_buttons)
                else:
                    pressed_str = "nomouse"

                # 📝 写入标签
                f.write(f"{frame} {x} {y} {pressed_str}\n")
                f.flush()  # 确保实时写入

                frame += 1

                # ⏱️ 控制帧率
                elapsed = tm.time() - start_time
                sleep_time = FRAME_DURATION - elapsed
                if sleep_time > 0:
                    tm.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n🛑 录制结束，共保存 {frame} 帧数据。")
        except Exception as e:
            print(f"❌ 录制出错: {e}")
        finally:
            listener.stop()