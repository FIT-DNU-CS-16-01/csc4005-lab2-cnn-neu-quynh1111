from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_CLASS_ORDER = [
    'Crazing',
    'Inclusion',
    'Patches',
    'Pitted_Surface',
    'Rolled-in_Scale',
    'Scratches',
]
CLASS_ALIASES = {
    'crazing': 'Crazing',
    'inclusion': 'Inclusion',
    'patches': 'Patches',
    'pitted_surface': 'Pitted_Surface',
    'pitted-surface': 'Pitted_Surface',
    'rolled-in_scale': 'Rolled-in_Scale',
    'rolled_in_scale': 'Rolled-in_Scale',
    'rolled-in-scale': 'Rolled-in_Scale',
    'scratches': 'Scratches',
}
FILENAME_LABEL_PATTERN = re.compile(r'([A-Za-z_-]+)_\d+', re.IGNORECASE)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    resolved_data_dir: str


class ImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: Callable | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ImageTensorTransform:
    def __init__(
        self,
        img_size: int,
        augment: bool = False,
        num_channels: int = 1,
        normalization: str = 'none',
    ):
        self.img_size = img_size
        self.augment = augment
        self.num_channels = num_channels
        self.normalization = normalization

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.img_size, self.img_size))
        if self.augment:
            angle = float(np.random.uniform(-8, 8))
            image = image.rotate(angle)
            dx = int(np.random.uniform(-0.05, 0.05) * self.img_size)
            dy = int(np.random.uniform(-0.05, 0.05) * self.img_size)
            canvas = Image.new('L', (self.img_size, self.img_size), color=0)
            canvas.paste(image, (dx, dy))
            image = canvas
            brightness = float(np.random.uniform(0.9, 1.1))
            contrast = float(np.random.uniform(0.9, 1.1))
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        arr = np.asarray(image, dtype=np.float32) / 255.0  # H x W
        if self.num_channels == 1:
            arr = arr[None, :, :]
        elif self.num_channels == 3:
            arr = np.stack([arr, arr, arr], axis=0)
        else:
            raise ValueError('num_channels chỉ hỗ trợ 1 hoặc 3.')

        if self.normalization == 'imagenet':
            arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        elif self.normalization != 'none':
            raise ValueError(f'Unsupported normalization: {self.normalization}')

        return torch.from_numpy(arr.astype(np.float32))


def _normalize_label_name(name: str) -> str | None:
    key = name.strip().lower().replace(' ', '_')
    return CLASS_ALIASES.get(key)


def _ordered_class_names(names: list[str]) -> list[str]:
    unique = sorted(set(names))
    if set(unique).issubset(set(DEFAULT_CLASS_ORDER)):
        return [name for name in DEFAULT_CLASS_ORDER if name in unique]
    return unique


def _extract_zip_if_needed(data_path: Path) -> Path:
    if data_path.is_dir():
        return data_path
    if data_path.is_file() and data_path.suffix.lower() == '.zip':
        extract_root = data_path.parent / f'{data_path.stem}_extracted'
        marker = extract_root / '.extracted_ok'
        if not marker.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(extract_root)
            marker.write_text('ok', encoding='utf-8')
        return extract_root
    raise FileNotFoundError(f'Không tìm thấy dữ liệu tại: {data_path}')


def _scan_class_folders(root: Path) -> tuple[list[tuple[Path, int]], list[str]] | None:
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not class_dirs:
        return None
    normalized: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        class_name = _normalize_label_name(class_dir.name)
        if class_name is None:
            return None
        normalized.append((class_dir, class_name))
    class_names = _ordered_class_names([name for _, name in normalized])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []
    for class_dir, class_name in normalized:
        for path in sorted(class_dir.rglob('*')):
            if path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((path, class_to_idx[class_name]))
    return (samples, class_names) if samples else None


def _infer_label_from_filename(path: Path) -> str | None:
    match = FILENAME_LABEL_PATTERN.match(path.stem)
    if not match:
        return None
    return _normalize_label_name(match.group(1))


def _scan_flat_images(root: Path) -> tuple[list[tuple[Path, int]], list[str]] | None:
    image_paths = [p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    if not image_paths:
        return None
    labels_by_path: list[tuple[Path, str]] = []
    for path in image_paths:
        class_name = _infer_label_from_filename(path)
        if class_name is not None:
            labels_by_path.append((path, class_name))
    if not labels_by_path:
        return None
    class_names = _ordered_class_names([class_name for _, class_name in labels_by_path])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = [(path, class_to_idx[class_name]) for path, class_name in labels_by_path]
    return samples, class_names


def _find_split_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = {n.lower() for n in names}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child
    return None


def _find_images_dir(split_dir: Path) -> Path | None:
    # Common patterns in NEU-CLS starter datasets:
    # - train/images
    # - train/train/images
    # - valid/images
    # - valid/valid/images
    direct = split_dir / 'images'
    if direct.is_dir():
        return direct

    nested_same = split_dir / split_dir.name / 'images'
    if nested_same.is_dir():
        return nested_same

    for child in split_dir.iterdir():
        if child.is_dir():
            maybe = child / 'images'
            if maybe.is_dir():
                return maybe
    return None


def _collect_labeled_images(root: Path) -> list[tuple[Path, str]]:
    # Try "class folders" first.
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    normalized: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        class_name = _normalize_label_name(class_dir.name)
        if class_name is None:
            normalized = []
            break
        normalized.append((class_dir, class_name))
    if normalized:
        labeled: list[tuple[Path, str]] = []
        for class_dir, class_name in normalized:
            for path in sorted(class_dir.rglob('*')):
                if path.suffix.lower() in IMG_EXTENSIONS:
                    labeled.append((path, class_name))
        if labeled:
            return labeled

    # Fallback: infer labels from file name like "crazing_10.jpg".
    image_paths = [p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    labeled = []
    for path in image_paths:
        class_name = _infer_label_from_filename(path)
        if class_name is None:
            continue
        labeled.append((path, class_name))
    return labeled


def _resolve_folder_splits(
    root: Path,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]], list[str]] | None:
    train_dir = _find_split_dir(root, ('train',))
    val_dir = _find_split_dir(root, ('valid', 'val'))
    test_dir = _find_split_dir(root, ('test',))
    if train_dir is None or val_dir is None:
        return None

    train_images = _find_images_dir(train_dir) or train_dir
    val_images = _find_images_dir(val_dir) or val_dir
    test_images = (_find_images_dir(test_dir) or test_dir) if test_dir is not None else None

    train_labeled = _collect_labeled_images(train_images)
    val_labeled = _collect_labeled_images(val_images)
    test_labeled = _collect_labeled_images(test_images) if test_images is not None else []

    if not train_labeled or not val_labeled:
        return None

    all_names = [name for _, name in (train_labeled + val_labeled + test_labeled)]
    class_names = _ordered_class_names(all_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_samples = [(p, class_to_idx[name]) for p, name in train_labeled]

    if test_labeled:
        val_samples = [(p, class_to_idx[name]) for p, name in val_labeled]
        test_samples = [(p, class_to_idx[name]) for p, name in test_labeled]
        return train_samples, val_samples, test_samples, class_names

    # If there is no explicit test folder, split the provided validation folder into val/test.
    if test_size <= 0:
        val_samples = [(p, class_to_idx[name]) for p, name in val_labeled]
        return train_samples, val_samples, val_samples, class_names

    # Heuristic: some NEU-CLS packages provide an extremely small "valid" split (e.g., 5 images/class).
    # In that case, keep valid as validation, and create a reasonable test split from the train set.
    if len(val_labeled) < 50:
        train_labels = [class_to_idx[name] for _, name in train_labeled]
        train_paths = [str(p) for p, _ in train_labeled]
        new_train_paths, test_paths, new_train_labels, test_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=test_size,
            stratify=train_labels,
            random_state=random_state,
        )
        new_train_samples = [(Path(p), y) for p, y in zip(new_train_paths, new_train_labels)]
        test_samples = [(Path(p), y) for p, y in zip(test_paths, test_labels)]
        val_samples = [(p, class_to_idx[name]) for p, name in val_labeled]
        return new_train_samples, val_samples, test_samples, class_names

    labels = [class_to_idx[name] for _, name in val_labeled]
    paths = [str(p) for p, _ in val_labeled]
    relative_test_size = test_size / max(val_size + test_size, 1e-8)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=relative_test_size,
        stratify=labels,
        random_state=random_state,
    )
    val_samples = [(Path(p), y) for p, y in zip(val_paths, val_labels)]
    test_samples = [(Path(p), y) for p, y in zip(test_paths, test_labels)]
    return train_samples, val_samples, test_samples, class_names


def _resolve_samples(data_dir: str | Path) -> tuple[list[tuple[Path, int]], list[str], Path]:
    root = _extract_zip_if_needed(Path(data_dir))
    direct_scan = _scan_class_folders(root)
    if direct_scan is not None:
        samples, class_names = direct_scan
        return samples, class_names, root
    flat_scan = _scan_flat_images(root)
    if flat_scan is not None:
        samples, class_names = flat_scan
        return samples, class_names, root
    raise ValueError(
        'Không đọc được dữ liệu. Hỗ trợ 2 kiểu: '
        '(1) thư mục lớp Crazing/...; '
        '(2) thư mục phẳng hoặc .zip có tên ảnh dạng crazing_10.jpg, inclusion_20.jpg, ...'
    )


def create_dataloaders(
    data_dir: str | Path,
    img_size: int = 64,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment: bool = False,
    num_workers: int = 0,
    num_channels: int = 1,
    normalization: str = 'none',
) -> SplitData:
    resolved_root = _extract_zip_if_needed(Path(data_dir))
    folder_split = _resolve_folder_splits(
        resolved_root,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    if folder_split is not None:
        train_samples, val_samples, test_samples, class_names = folder_split
    else:
        samples, class_names, resolved_root = _resolve_samples(resolved_root)
        paths = [str(p) for p, _ in samples]
        labels = [label for _, label in samples]

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths,
            labels,
            test_size=val_size + test_size,
            stratify=labels,
            random_state=random_state,
        )
        relative_test_size = test_size / (val_size + test_size)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=relative_test_size,
            stratify=temp_labels,
            random_state=random_state,
        )

        train_samples = [(Path(p), y) for p, y in zip(train_paths, train_labels)]
        val_samples = [(Path(p), y) for p, y in zip(val_paths, val_labels)]
        test_samples = [(Path(p), y) for p, y in zip(test_paths, test_labels)]

    train_ds = ImageDataset(
        train_samples,
        transform=ImageTensorTransform(img_size, augment=augment, num_channels=num_channels, normalization=normalization),
    )
    eval_tf = ImageTensorTransform(img_size, augment=False, num_channels=num_channels, normalization=normalization)
    val_ds = ImageDataset(val_samples, transform=eval_tf)
    test_ds = ImageDataset(test_samples, transform=eval_tf)

    return SplitData(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=class_names,
        resolved_data_dir=str(resolved_root),
    )
