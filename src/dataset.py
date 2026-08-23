from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from configs.classes import CLASSES, CLASS_TO_IDX
from configs.config import BATCH_SIZE, NUM_WORKERS, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.transforms import build_transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class WasteDataset(Dataset):
    """data/<split>/<class_name>/*.jpg 구조를 읽는다.

    torchvision ImageFolder 대신 직접 구현해 클래스 인덱스가
    configs/classes.py의 순서와 항상 일치하도록 고정한다.
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_name in CLASSES:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMG_EXTS:
                    self.samples.append((path, CLASS_TO_IDX[class_name]))

        if not self.samples:
            raise RuntimeError(f"이미지를 찾지 못했습니다: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self):
        counts = {name: 0 for name in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts


def build_dataloaders(augment=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = WasteDataset(TRAIN_DIR, build_transforms(train=True, augment=augment))
    val_ds = WasteDataset(VAL_DIR, build_transforms(train=False))

    # worker를 쓸 때만 지속/프리페치 옵션이 유효하다
    extra = {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False, **extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, **extra,
    )
    return train_loader, val_loader


def build_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    test_ds = WasteDataset(TEST_DIR, build_transforms(train=False))
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
