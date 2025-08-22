# model.py
# 项目: AIusePC - 让AI帮你用PC
# 用途: 定义AI模型结构，用于预测鼠标位置和点击状态
# GitHub: github.com/imjumping

import torch
import torch.nn as nn
import torchvision.models as models

class MousePredictor(nn.Module):
    """
    使用预训练 ResNet18 作为主干网络
    输出两个结果：
      1. 鼠标坐标 (x, y) 归一化到 [0,1]
      2. 是否有按键按下 (0=无, 1=有)
    """
    def __init__(self):
        super(MousePredictor, self).__init__()

        # 使用 ResNet18 作为视觉特征提取器
        base_model = models.resnet18(pretrained=True)  # 使用 ImageNet 预训练权重

        # 去掉最后的分类层（fc），保留前面的卷积特征
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # 冻结部分底层参数（可选，加快训练）
        # for param in self.backbone[:7].parameters():
        #     param.requires_grad = False

        # 回归头：预测归一化坐标 (x, y)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()  # 强制输出在 [0,1] 范围内
        )

        # 分类头：预测是否有鼠标点击（二分类）
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        """
        输入: 图像张量 (B, 3, 224, 224)，范围 [0,1]
        输出:
            - pred_coords: (B, 2)  归一化坐标
            - pred_click:  (B,)    是否点击（0.~1.）
        """
        # 提取图像特征
        features = self.backbone(x)  # 输出形状: (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # 展平成 (B, 512)

        # 预测坐标
        pred_coords = self.regressor(features)

        # 预测点击
        pred_click = self.classifier(features).squeeze(-1)  # 去掉最后一个维度

        return pred_coords, pred_click


# ==========================
# 🧪 使用示例（调试用）
# ==========================
if __name__ == "__main__":
    print("🧪 正在测试 model.py...")

    # 创建模型
    model = MousePredictor()
    print("✅ 模型创建成功！")

    # 模拟输入：1张 224x224 RGB 图像
    dummy_input = torch.randn(1, 3, 224, 224)

    # 前向传播
    with torch.no_grad():
        coords, click = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出坐标: {coords.shape} -> {coords.numpy().squeeze()}")
    print(f"输出点击: {click.shape} -> {click.item():.3f} (点击概率)")

    print("🎉 模型测试通过！")