from torchvision import transforms

from configs.config import IMG_SIZE, MEAN, STD


def build_transforms(train: bool, augment: bool = True):
    """train=True면 학습용, False면 평가/추론용 변환을 만든다.

    augment=False로 두면 Baseline(증강 없음) 실험을 재현할 수 있다.
    """
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
