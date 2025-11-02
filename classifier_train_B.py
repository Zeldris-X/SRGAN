import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def main():
  
    DATA_ROOT = "data/srgan_generated"     
    SAVE_DIR = "results/classifier_B"
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 2  

   
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f" Loaded SRGAN dataset → {len(train_ds)} train, {len(test_ds)} test, classes: {train_ds.classes}")

   
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False 

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.4f}  Train Acc: {train_acc:.2f}%")

    
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs)

    report = classification_report(y_true, y_pred, target_names=train_ds.classes, digits=3)
    auc = roc_auc_score(y_true, y_prob)
    print("\nClassification Report:\n", report)
    print(f"AUC: {auc:.3f}")

    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_ds.classes, yticklabels=train_ds.classes)
    plt.title("Model B (MobileNetV2 on SRGAN Images)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_B.png"))
    plt.close()

    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "classifier_B.pth"))
    with open(os.path.join(SAVE_DIR, "report_B.txt"), "w") as f:
        f.write(report + f"\nAUC: {auc:.3f}\n")

    print(f"\n Model B saved to {SAVE_DIR}")



if __name__ == "__main__":
    main()
