# 基于 MobileNetV2 的生活垃圾图像分类

深圳大学 · 人工智能课程实验项目

## 项目简介
本项目基于 PyTorch，使用轻量级卷积神经网络 MobileNetV2，通过迁移学习完成生活垃圾图像分类任务。
项目可在普通 CPU 上训练与推理，用于课程实验与小型识别场景。

## 数据集说明
包含 6 类生活垃圾：
- glass（玻璃）
- paper（纸张）
- cardboard（纸板）
- plastic（塑料）
- metal（金属）
- trash（其他垃圾）

数据划分：
- 训练集：1768 张
- 验证集：328 张
- 测试集：431 张

## 上传文件说明

### 1. 代码文件
- train.py：模型训练代码。
  加载数据集、构建 MobileNetV2、执行 15 轮训练并保存最优模型。

- infer.py：单张图片推理代码。
  加载训练好的模型，输入图片输出分类结果。

### 2. 模型文件
- best_garbage_model.pth：训练后保存的最优权重文件。

### 3. 测试图片
- 奥利奥.jpg：纸板类测试图片。
- 玻璃.jpg：玻璃类测试图片。

### 4. 运行结果截图
- infer_cardboard.jpg：奥利奥图片推理终端输出。
- infer_glass.jpg：玻璃图片推理终端输出。

## 环境依赖
```bash
pip install torch torchvision pillow

运行方法
1. 训练模型

运行python train.py
训练 15 轮，自动保存最优模型 best_garbage_model.pth。
2. 推理预测

运行python infer.py
默认读取 奥利奥.jpg，输出分类结果。
模型结构
主干网络：MobileNetV2（预训练权重冻结）
分类头：6 类全连接层
优化器：Adam，学习率 0.001
损失函数：交叉熵损失
实验结果
训练准确率：94%–98%
验证准确率：89%–92%
测试准确率：90%–93%
纸板、纸张、塑料识别精度最高；玻璃、金属因反光特征相似，存在少量混淆。
测试示例
奥利奥包装盒 → cardboard（纸板），结果正确
玻璃罐头 → glass（玻璃），结果正确
