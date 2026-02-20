"""Data augmentation and preprocessing pipelines."""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline.

    Order matters: spatial → color → tensor conversion → normalization → erasing.

    Args:
        image_size: Target image size (square). Default: 224

    Returns:
        Composed transform pipeline for training data.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        # Bridge domain gap: sun/shade/overcast variation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        # Simulate out-of-focus mobile photos
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        # Simulate leaf occlusion (insects, fingers, overlap)
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/Test preprocessing pipeline.

    No augmentation — deterministic for reproducible evaluation.

    Args:
        image_size: Target image size (square). Default: 224

    Returns:
        Composed transform pipeline for validation/test data.
    """
    resize_size = int(image_size * 1.14)  # 256 for 224
    return transforms.Compose([
        transforms.Resize(size=resize_size),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
