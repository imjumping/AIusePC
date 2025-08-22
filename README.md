# AIusePC | 让AI帮你用PC

首先，你需要在dataset文件夹下放入你自己的数据集（通常是视频和标注鼠标移动的位置）

数据集格式：
视频放在dataset/video.mp4，
鼠标标注数据：放在dataset/mouse_data/data.txt

## 鼠标标注示例格式（每帧标注一次鼠标位置）
#格式：时间 鼠标{x y} 左右键
```txt
1 0 0 button.left
2 1 0 button.right
3 2 1 nomouse
4 3 2 nomouse
5 12 12 nomouse
6 12 12 button.left
7 12 12 button.middle
8 12 12 button.middle, button.right
…
```
上面示例代码表示：
1. 第1帧，鼠标位置在(0,0)，左键点击
2. 第2帧，鼠标位置在(1,0)，右键点击
3. 第3帧，鼠标位置在(2,1)，无点击
4. 第4帧，鼠标位置在(3,2)，无点击
5. 第5帧，鼠标位置在(12,12)，无点击
6. 第6帧，鼠标位置在(12,12)，左键点击
7. 第7帧，鼠标位置在(12,12)，中键点击
8. 第8帧，鼠标位置在(12,12)，中键点击，并右键点击
…
这个格式较为容易理解，也比较方便。
---
## Get start!
打开train.py

点击开始，开始训练

训练完成生成model文件夹，并会在model/目录创建一个model.bin文件

## 注意事项
1. 本人12岁，可能做的不太好，欢迎指正
2. 处理依赖：pyautogui,python3.13,numpy,opencv-python,pynput

github.com/imjumping

请遵守协议：MIT