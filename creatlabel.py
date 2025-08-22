import pyautogui as pg
import time as tm
import sys
from pynput import mouse
import os

FPS = 30
FRAMEDURATION = 1.0 / FPS

currentlypressedbuttons = set()

def onclick(x, y, button, pressed):
    global currentlypressedbuttons
    if pressed:
        currentlypressedbuttons.add(button)
    else:
        currentlypressedbuttons.discard(button)

listener = mouse.Listener(on_click=onclick)
listener.start()

frame = 0
print("程序已启动，按 Ctrl+C 退出。")

# 确保目录存在
directory = "./dataset/mouse_data"
if not os.path.exists(directory):
    os.makedirs(directory)

try:
    with open(os.path.join(directory, "data.txt"), "w") as f:
        while True:
            loopstarttime = tm.time()
            x, y = pg.position()
            if currentlypressedbuttons:
                pressedstr = ", ".join([str(b) for b in currentlypressedbuttons])
            else:
                pressedstr = "nomouse"
            f.write(f"{frame} {x} {y} {pressedstr}\n")
            f.flush()
            frame += 1
            elapsedtime = tm.time() - loopstarttime
            sleepduration = FRAMEDURATION - elapsedtime
            if sleepduration > 0:
                tm.sleep(sleepduration)
except KeyboardInterrupt:
    print("\n程序已停止。")
finally:
    if listener.is_alive():  # 注意：是 is_alive() 而不是 isalive()
        listener.stop()