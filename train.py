import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm

from data.pets_dataset import PetsDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


# ========================
# TRANSFORMS
# ========================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ========================
# CLASSIFIER TRAINING
# ========================
def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PetsDataset(root=args.data_dir, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VGG11Classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        total_correct = 0
        total = 0

        for images, labels, _, _ in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = total_correct / total
        print(f"Epoch {epoch+1}: Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "classifier.pth"))


# ========================
# LOCALIZER TRAINING
# ========================
def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PetsDataset(root=args.data_dir, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = VGG11Localizer().to(device)

    # ✅ FIX: better loss
    reg_loss = nn.SmoothL1Loss()
    iou = IoULoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    # 🔥 helper function (define once, not inside loop)
    def cxcywh_to_xyxy(box):
        cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images, _, bboxes, _ in tqdm(loader):
            images = images.to(device)
            bboxes = bboxes.to(device)

            preds = model(images)

            # ✅ FIX: normalize for stable training
            preds_norm = preds / 224.0
            bboxes_norm = bboxes / 224.0

            # ✅ FIX: convert format for IoU
            pred_xy = cxcywh_to_xyxy(preds_norm)
            gt_xy   = cxcywh_to_xyxy(bboxes_norm)

            # ✅ FINAL LOSS
            loss = reg_loss(preds_norm, bboxes_norm) + iou(pred_xy, gt_xy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "localizer.pth"))

# ========================
# SEGMENTATION TRAINING
# ========================
def train_segmenter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PetsDataset(root=args.data_dir, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VGG11UNet(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images, _, _, masks in tqdm(loader):
            images = images.to(device)

            # Convert mask to tensor + class indices
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "unet.pth"))


# ========================
# MAIN
# ========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--task", type=str, default="all")

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.task in ["classifier", "all"]:
        train_classifier(args)

    if args.task in ["localizer", "all"]:
        train_localizer(args)

    if args.task in ["segmenter", "all"]:
        train_segmenter(args)


if __name__ == "__main__":
    main()