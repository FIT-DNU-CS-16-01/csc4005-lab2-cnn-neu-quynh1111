from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import (
    EarlyStopping,
    classification_report_dict,
    compute_accuracy,
    count_parameters,
    ensure_dir,
    plot_curves,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    set_seed,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train CNN for NEU surface defect classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới NEU-CLS.zip hoặc thư mục dữ liệu đã giải nén')
    parser.add_argument('--project', type=str, default='csc4005-lab2-neu-cnn')
    parser.add_argument('--run_name', type=str, default='debug_run')
    parser.add_argument('--model_name', type=str, choices=['cnn_small', 'resnet18', 'mobilenet_v2', 'vgg11_bn'], default='cnn_small')
    parser.add_argument('--train_mode', type=str, choices=['scratch', 'transfer', 'finetune'], default='scratch')
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--scheduler', type=str, choices=['none', 'plateau'], default='plateau')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_name == 'cnn_small' and args.train_mode != 'scratch':
        raise ValueError('`cnn_small` chỉ dùng với `--train_mode scratch`.')
    if args.model_name != 'cnn_small' and args.train_mode == 'scratch':
        raise ValueError('Backbone pretrained phải đi với `--train_mode transfer` hoặc `--train_mode finetune`.')


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def resolve_input_mode(args: argparse.Namespace) -> tuple[int, str]:
    if args.train_mode == 'scratch':
        return 1, 'none'
    return 3, 'imagenet'


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), y_true, y_pred


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = ensure_dir(Path('outputs') / args.run_name)
    num_channels, normalization = resolve_input_mode(args)

    data = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment=args.augment,
        num_workers=args.num_workers,
        num_channels=num_channels,
        normalization=normalization,
    )
    print(f'Resolved data directory: {data.resolved_data_dir}')
    print(f'Classes: {data.class_names}')
    split_sizes = {
        'n_train': len(data.train_loader.dataset),
        'n_val': len(data.val_loader.dataset),
        'n_test': len(data.test_loader.dataset),
    }

    model = build_model(
        model_name=args.model_name,
        train_mode=args.train_mode,
        num_classes=len(data.class_names),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if args.scheduler == 'plateau' else None
    total_params, trainable_params = count_parameters(model)

    use_wandb = args.use_wandb and wandb is not None
    if args.use_wandb and wandb is None:
        print('Cảnh báo: chưa import được wandb. Chạy tiếp ở chế độ không log online.')
    if use_wandb:
        try:
            wandb.init(project=args.project, name=args.run_name, config=vars(args))
        except Exception as exc:  # pragma: no cover - env/network/auth dependent
            print(f'W&B init failed ({type(exc).__name__}). Fallback to offline mode...')
            try:
                try:
                    wandb.finish()
                except Exception:
                    pass
                wandb.init(project=args.project, name=args.run_name, config=vars(args), mode='offline')
            except Exception as exc2:  # pragma: no cover
                print(f'W&B offline init also failed ({type(exc2).__name__}). Disable W&B logging for this run.')
                use_wandb = False
        if use_wandb:
            wandb.config.update({
                'num_classes': len(data.class_names),
                'class_names': data.class_names,
                'device': str(device),
                'resolved_data_dir': data.resolved_data_dir,
                **split_sizes,
                'num_channels': num_channels,
                'normalization': normalization,
                'total_params': total_params,
                'trainable_params': trainable_params,
            })

    history: list[dict[str, float]] = []
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_time = time.perf_counter() - start
        lr_current = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': lr_current,
            'epoch_time_sec': round(epoch_time, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_current:.6f} | sec={epoch_time:.2f}"
        )
        if use_wandb:
            wandb.log(row)
        if early_stopper.step(val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        if early_stopper.should_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    if not (output_dir / 'best_model.pt').exists():
        torch.save(model.state_dict(), output_dir / 'best_model.pt')

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, data.test_loader, criterion, device)
    report = classification_report_dict(y_true, y_pred, data.class_names)
    cm = save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / 'confusion_matrix.png')
    plot_curves(history, output_dir / 'curves.png')
    save_history_csv(history, output_dir / 'history.csv')
    avg_epoch_time = sum(row['epoch_time_sec'] for row in history) / max(len(history), 1)
    metrics = {
        'model_name': args.model_name,
        'train_mode': args.train_mode,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'avg_epoch_time_sec': avg_epoch_time,
        **split_sizes,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'class_names': data.class_names,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'resolved_data_dir': data.resolved_data_dir,
        'normalization': normalization,
        'num_channels': num_channels,
    }
    save_json(metrics, output_dir / 'metrics.json')
    print(f'Best val acc: {best_val_acc:.4f}')
    print(f'Test acc: {test_acc:.4f}')
    print(f'Average epoch time: {avg_epoch_time:.2f} sec')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Saved outputs to: {output_dir}')

    if use_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'avg_epoch_time_sec': avg_epoch_time,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'confusion_matrix_image': wandb.Image(str(output_dir / 'confusion_matrix.png')),
            'curves_image': wandb.Image(str(output_dir / 'curves.png')),
        })
        wandb.finish()


if __name__ == '__main__':
    main()
