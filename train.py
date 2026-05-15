import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

data_root = r"D:\人工智能\rubbish分类\pythonProject\数据集\Garbage classification"

train_txt = r"D:\人工智能\rubbish分类\pythonProject\数据集\one-indexed-files-notrash_train.txt"
val_txt   = r"D:\人工智能\rubbish分类\pythonProject\数据集\one-indexed-files-notrash_val.txt"
test_txt  = r"D:\人工智能\rubbish分类\pythonProject\数据集\one-indexed-files-notrash_test.txt"

batch_size = 32
epochs = 15
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2class = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

class GarbageDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, label = line.split()
                label = int(label) - 1
                cls_name = id2class[label]
                img_path = os.path.join(data_root, cls_name, img_name)

                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = GarbageDataset(train_txt, train_transform)
val_dataset   = GarbageDataset(val_txt, val_test_transform)
test_dataset  = GarbageDataset(test_txt, val_test_transform)

if len(train_dataset) == 0:
    print("错误：没有读到任何图片！")
    exit()

print("训练集: {}, 验证集: {}, 测试集: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, 6)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

best_acc = 0.0
print("\n开始训练...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels)

    train_loss /= len(train_dataset)
    train_acc = train_correct.float() / len(train_dataset)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels)

    val_acc = val_correct.float() / len(val_dataset)

    print("Epoch {} | 训练损失: {:.3f} | 训练准确率: {:.2%} | 验证准确率: {:.2%}".format(
        epoch+1, train_loss, train_acc, val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_garbage_model.pth")
        print("已保存最优模型 | 最佳验证准确率: {:.2%}".format(best_acc))

print("\n训练完成，开始测试...")
model.load_state_dict(torch.load("best_garbage_model.pth"))
model.eval()

test_correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels)

test_acc = test_correct.float() / len(test_dataset)
print("\n最终测试集准确率: {:.2%}".format(test_acc))