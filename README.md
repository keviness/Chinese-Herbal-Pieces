# 基于ResNet18的中药植物及饮片图片分类说明书

## 1. 项目简介

本项目基于ResNet18深度卷积神经网络，实现中药植物及饮片图片的自动分类。适用于中药材识别、辅助教学等场景。

## 2. 环境配置

建议使用如下环境：

- Python >= 3.7
- PyTorch >= 1.8
- torchvision >= 0.9
- CUDA（可选，建议有GPU加速）

安装依赖：

```bash
pip install torch torchvision
```

## 3. 数据准备

1. 数据集应按如下结构组织，每个类别一个文件夹，文件夹名为类别名：
   ```
   dataset/
     ├── train/
     │    ├── 类别A/
     │    └── 类别B/
     └── val/
          ├── 类别A/
          └── 类别B/
   ```
2. 图片建议为jpg或png格式，分辨率建议不低于224x224。

## 4. 训练流程

1. 加载数据集，使用 `torchvision.datasets.ImageFolder`。
2. 数据增强建议使用 `transforms.RandomResizedCrop(224)`、`transforms.RandomHorizontalFlip()`等。
3. 加载ResNet18模型，可用 `torchvision.models.resnet18(pretrained=True)`，并根据类别数修改最后的全连接层：
   ```python
   import torchvision.models as models
   model = models.resnet18(pretrained=True)
   model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
   ```
4. 设置损失函数（如 `CrossEntropyLoss`）和优化器（如 `Adam`或 `SGD`）。
5. 训练模型，保存最佳权重。

## 5. 推理与评估

1. 加载训练好的模型权重。
2. 对单张图片进行预处理（缩放、归一化等），送入模型预测。
3. 输出类别概率或标签。

## 6. 示例代码片段

```python
from torchvision import transforms, models
from PIL import Image
import torch

# 加载模型
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = Image.open('test.jpg')
img = transform(img).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)
    print('预测类别:', pred.item())
```

## 7. 注意事项

- 数据集需保证类别均衡，图片清晰。
- 可根据实际需求调整模型结构和参数。
- 若类别较多，建议使用更深层的ResNet或其他模型。

## 8. 参考文献

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- PyTorch官方文档：https://pytorch.org/docs/stable/torchvision/models.html
