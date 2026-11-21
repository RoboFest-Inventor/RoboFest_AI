import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

BATCH_SIZE = 32
EPOCHS = 30
INPUT_SIZE = 224

with open("config.json", "r") as f:
    patch_size = json.load(f).get("patch_size", 300)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"학습 장치: {device} | 설정된 상자 크기: {patch_size}px")

train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(35),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_base = datasets.ImageFolder("dataset", transform=train_transform)
val_base = datasets.ImageFolder("dataset", transform=val_transform)
class_names = train_base.classes
print(f"클래스: {class_names} | 전체 학습 사진: {len(train_base)}장")

indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(42)).tolist()
val_size = max(1, int(len(indices) * 0.2))
train_ds = Subset(train_base, indices[val_size:])
val_ds = Subset(val_base, indices[:val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.1f}%")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "classes": class_names,
            "patch_size": patch_size,
            "input_size": INPUT_SIZE
        }, "lego_classifier.pth")

print(f"\n학습 완료! 최고 검증 정확도: {best_acc:.1f}% (lego_classifier.pth 저장됨)")
