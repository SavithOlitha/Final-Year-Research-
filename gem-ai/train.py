import os, random
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

DATA_DIR = Path("composites")  # composites/real, composites/synthetic
IMG_SIZE = 224
BATCH = 16
EPOCHS = 10
LR = 1e-4

class SimpleFolderDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def load_items():
    items = []
    for label_name, label_id in [("real", 0), ("synthetic", 1)]:
        folder = DATA_DIR / label_name
        for p in folder.glob("*.jpg"):
            items.append((str(p), label_id))
    return items

def main():
    items = load_items()
    if len(items) < 20:
        print("Not enough data. Try at least 20+ composites (better 100+).")
        return

    random.shuffle(items)
    train_items, val_items = train_test_split(items, test_size=0.2, random_state=42)

    tfm_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = SimpleFolderDataset(train_items, tfm_train)
    val_ds   = SimpleFolderDataset(val_items, tfm_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 2 classes
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss:.3f} | val_acc={acc:.3f}")

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), "gem_model.pt")
            print("Saved: gem_model.pt")

    print("Best val acc:", best_val)

if __name__ == "__main__":
    main()
