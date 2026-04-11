import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from models.multitask import MultiTaskPerceptionModel


# ========================
# TRANSFORM
# ========================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ========================
# LOAD IMAGE
# ========================
def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = get_transform()
    return transform(image).unsqueeze(0)


# ========================
# LOAD MODEL + WEIGHTS
# ========================
def load_model(args, device):
    model = MultiTaskPerceptionModel(
        num_breeds=args.num_classes,
        seg_classes=args.seg_classes,
        in_channels=3
    )

    # Load weights manually
    ckpt_dir = args.checkpoint_dir

    try:
        model.classifier_head.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "classifier.pth"), map_location=device)
        )
        print("Loaded classifier weights")
    except:
        print("Warning: classifier weights not found")

    try:
        model.localization_head.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "localizer.pth"), map_location=device)
        )
        print("Loaded localizer weights")
    except:
        print("Warning: localizer weights not found")

    try:
        model.segmenter.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "unet.pth"), map_location=device)
        )
        print("Loaded segmenter weights")
    except:
        print("Warning: segmenter weights not found")

    model.to(device)
    model.eval()

    return model


# ========================
# INFERENCE
# ========================
def infer(image_path, model, device):
    image = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)

    # ========================
    # CLASSIFICATION
    # ========================
    probs = torch.softmax(outputs["classification"][0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)

    print("\n[CLASSIFICATION]")
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
        print(f"{i}. Class {idx.item()} → {prob.item():.4f}")

    # ========================
    # LOCALIZATION
    # ========================
    bbox = outputs["localization"][0]

    x, y, w, h = bbox.tolist()

    # Clamp values (important)
    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(224, int(x + w / 2))
    y2 = min(224, int(y + h / 2))

    print("\n[LOCALIZATION]")
    print(f"Center: ({x:.2f}, {y:.2f})")
    print(f"Box: ({x1}, {y1}) → ({x2}, {y2})")

    # ========================
    # SEGMENTATION
    # ========================
    seg = torch.argmax(outputs["segmentation"][0], dim=0)

    print("\n[SEGMENTATION]")
    print("Classes:", torch.unique(seg).tolist())

    return seg.cpu().numpy()


# ========================
# MAIN
# ========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_classes", type=int, default=37)
    parser.add_argument("--seg_classes", type=int, default=3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args, device)

    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    else:
        paths = [
            os.path.join(args.image_path, f)
            for f in os.listdir(args.image_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    for path in paths:
        print("\n" + "=" * 50)
        print(f"Processing: {path}")
        infer(path, model, device)

    print("\nDone!")


if __name__ == "__main__":
    main()