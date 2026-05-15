import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ===================== 【必须和训练完全一致】 =====================
data_root = r"D:\人工智能\rubbish分类\pythonProject\数据集\Garbage classification"
id2class = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 图片预处理（必须和验证集一样） =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================== 加载模型（必须和训练结构一致） =====================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load("best_garbage_model.pth", map_location=device))
model.to(device)
model.eval()

# ===================== 预测单张图片函数 =====================
def predict_image(img_path):
    if not os.path.exists(img_path):
        return "图片不存在"

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        class_name = id2class[pred.item()]

    return class_name

# ===================== 测试：这里改你要预测的图片路径 =====================
if __name__ == "__main__":
    # 你只需要改这一行！换成你的图片路径
    test_img = r"D:\人工智能\rubbish分类\pythonProject\奥利奥.jpg"

    result = predict_image(test_img)
    print("预测结果：", result)