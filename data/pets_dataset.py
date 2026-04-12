import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

# ================================
# DATASET URLS
# ================================
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

from torchvision import transforms

transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
# ================================
# DOWNLOAD + EXTRACT
# ================================
def download_and_extract(url, root):
    filename = url.split("/")[-1]
    filepath = os.path.join(root, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)

    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=root)


def prepare_dataset(root):
    if not os.path.exists(os.path.join(root, "images")):
        download_and_extract(IMAGES_URL, root)

    if not os.path.exists(os.path.join(root, "annotations")):
        download_and_extract(ANNOTATIONS_URL, root)


# ================================
# DATASET CLASS
# ================================
class PetsDataset(Dataset):
    def __init__(self, root="data", transform=transform):
        self.root = root
        self.transform = transform

        prepare_dataset(root)

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.bbox_dir = os.path.join(root, "annotations", "xmls")

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) if f.endswith(".jpg")
        ])

        # Build class mapping (37 breeds)
        self.class_to_idx = {}
        self._build_class_mapping()

    def _build_class_mapping(self):
        classes = set()
        for fname in self.image_files:
            breed = "_".join(fname.split("_")[:-1]).lower()
            classes.add(breed)
            

        classes = sorted(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
       

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # ========================
        # IMAGE
        # ========================
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # ========================
        # LABEL
        # ========================
        breed = "_".join(img_name.split("_")[:-1]).lower()
        label = self.class_to_idx[breed]

        # ========================
        # MASK
        # ========================
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path)

        # 🔥 FIX: resize mask
        mask = mask.resize((224, 224))

        # ========================
        # BBOX
        # ========================
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = os.path.join(self.bbox_dir, xml_name)

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bbox_xml = root.find("object").find("bndbox")

            xmin = int(bbox_xml.find("xmin").text)
            ymin = int(bbox_xml.find("ymin").text)
            xmax = int(bbox_xml.find("xmax").text)
            ymax = int(bbox_xml.find("ymax").text)

            orig_w, orig_h = image.size
            new_w, new_h = 224, 224

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            x_center = (xmin + xmax) / 2 * scale_x
            y_center = (ymin + ymax) / 2 * scale_y
            width = (xmax - xmin) * scale_x
            height = (ymax - ymin) * scale_y

            bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
        else:
            bbox = torch.tensor([112, 112, 50, 50], dtype=torch.float32)

        # ========================
        # TRANSFORM
        # ========================
        if self.transform:
            image = self.transform(image)

        # ========================
        # MASK PROCESSING
        # ========================
        mask = np.array(mask)

        mask[mask == 2] = 0
        mask[mask == 1] = 1
        mask[mask == 3] = 2

        mask = torch.tensor(mask, dtype=torch.long)

        return image, label, bbox, mask