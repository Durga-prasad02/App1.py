"""
Dataset Loader and Preprocessing Pipeline tailored for Municipal Waste Images
Includes:
- Domain-specific augmentation (rotation, color jitter, perspective distortion)
- PyTorch Dataset loader
- Synthetic sample generator for immediate testing without external dataset download

Author: AGGIBADALA VENKATA DURGA NAGA PRASAD
Guide: Ms. G. Hema Sri
"""

import os
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_waste_transforms(img_size: int = 224, is_train: bool = True):
    """
    Constructs domain-tailored image pre-processing and data augmentation pipelines.
    For waste recognition, bins introduce lighting shifts, rotational invariance, and crinkle distortion.
    """
    if not TORCH_AVAILABLE:
        return None

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


if TORCH_AVAILABLE:
    class MunicipalWasteDataset(Dataset):
        """
        PyTorch Dataset for Municipal Solid Waste images structured in class subfolders.
        """
        def __init__(self, root_dir: str, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.samples: List[Tuple[str, int]] = []
            self.classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            img_path, target = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, target


def create_synthetic_waste_sample(category: str, save_path: str, size: Tuple[int, int] = (300, 300)):
    """
    Generates a realistic synthetic waste sample image with visual attributes
    and save to disk. Enables instant testing of the Smart-Bin Assistant prototype.
    """
    img = Image.new("RGB", size, color="#F3F4F6")
    draw = ImageDraw.Draw(img)

    category_lower = category.lower()

    if "cardboard" in category_lower:
        # Brown corrugated carton box
        draw.rectangle([40, 60, 260, 240], fill="#C19A6B", outline="#8B5A2B", width=4)
        draw.rectangle([50, 70, 140, 120], fill="#B38B59")
        draw.line([40, 150, 260, 150], fill="#8B5A2B", width=3)  # Tape seam
        draw.rectangle([90, 170, 210, 210], fill="#E5D3B3")
        draw.text((105, 185), "RECYCLE", fill="#5C4033")

    elif "glass" in category_lower:
        # Translucent glass bottle
        draw.rectangle([130, 30, 170, 70], fill="#A7D7C5", outline="#5C8D89", width=3)  # Neck
        draw.polygon([(130, 70), (170, 70), (220, 120), (80, 120)], fill="#A7D7C5", outline="#5C8D89")
        draw.rectangle([80, 120, 220, 270], fill="#BFE3D5", outline="#5C8D89", width=4)
        draw.line([100, 140, 100, 250], fill="#FFFFFF", width=3)  # Specular highlight reflection
        draw.text((120, 180), "GLASS", fill="#285943")

    elif "metal" in category_lower:
        # Aluminum soda can
        draw.ellipse([90, 40, 210, 80], fill="#D1D5DB", outline="#4B5563", width=3)
        draw.rectangle([90, 60, 210, 240], fill="#9CA3AF", outline="#4B5563", width=3)
        draw.ellipse([90, 220, 210, 260], fill="#6B7280", outline="#4B5563", width=3)
        # Pull tab & label stripe
        draw.rectangle([90, 120, 210, 180], fill="#EF4444")
        draw.text((115, 142), "COLA CAN", fill="#FFFFFF")

    elif "paper" in category_lower:
        # Sheet of printed office paper / newspaper
        draw.polygon([(60, 40), (220, 40), (250, 70), (250, 260), (60, 260)], fill="#FFFFFF", outline="#9CA3AF", width=2)
        draw.polygon([(220, 40), (250, 70), (220, 70)], fill="#E5E7EB", outline="#9CA3AF", width=2)  # Folded corner
        for y in range(85, 240, 20):
            draw.line([80, y, 230, y], fill="#CBD5E1", width=3)
        draw.text((80, 50), "DAILY REPORT", fill="#1E293B")

    elif "plastic" in category_lower:
        # Plastic water bottle / detergent flask
        draw.rectangle([135, 30, 165, 55], fill="#3B82F6")  # Blue cap
        draw.polygon([(135, 55), (165, 55), (210, 95), (90, 95)], fill="#93C5FD", outline="#3B82F6", width=2)
        draw.rectangle([90, 95, 210, 250], fill="#BFDBFE", outline="#3B82F6", width=3)
        # Ridges & label
        draw.rectangle([90, 140, 210, 190], fill="#60A5FA")
        draw.text((115, 160), "PURE WATER", fill="#FFFFFF")
        draw.line([105, 110, 105, 230], fill="#FFFFFF", width=3)

    else:
        # Organic / Food waste (Apple core / Banana peel)
        draw.ellipse([80, 80, 220, 220], fill="#84CC16", outline="#4D7C0F", width=3)  # Apple
        draw.rectangle([145, 45, 155, 85], fill="#78350F")  # Stem
        draw.polygon([(155, 55), (195, 45), (175, 75)], fill="#15803D")  # Leaf
        draw.ellipse([110, 110, 140, 140], fill="#65A30D")  # Organic texture
        draw.text((105, 175), "FOOD WASTE", fill="#14532D")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    return save_path
