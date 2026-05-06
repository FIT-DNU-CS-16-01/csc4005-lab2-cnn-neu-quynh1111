from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_get(d: dict[str, Any], key: str, default: Any = '') -> Any:
    return d.get(key, default)


def load_runs(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not outputs_dir.exists():
        return rows

    for run_dir in sorted([p for p in outputs_dir.iterdir() if p.is_dir()]):
        metrics_path = run_dir / 'metrics.json'
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception:
            continue

        rows.append(
            {
                'run_name': run_dir.name,
                'model_name': _safe_get(metrics, 'model_name'),
                'train_mode': _safe_get(metrics, 'train_mode'),
                'best_epoch': _safe_get(metrics, 'best_epoch'),
                'best_val_acc': _safe_get(metrics, 'best_val_acc'),
                'best_val_loss': _safe_get(metrics, 'best_val_loss'),
                'test_acc': _safe_get(metrics, 'test_acc'),
                'test_loss': _safe_get(metrics, 'test_loss'),
                'avg_epoch_time_sec': _safe_get(metrics, 'avg_epoch_time_sec'),
                'trainable_params': _safe_get(metrics, 'trainable_params'),
                'total_params': _safe_get(metrics, 'total_params'),
                'normalization': _safe_get(metrics, 'normalization'),
                'num_channels': _safe_get(metrics, 'num_channels'),
                'resolved_data_dir': _safe_get(metrics, 'resolved_data_dir'),
            }
        )

    def sort_key(row: dict[str, Any]):
        try:
            return float(row.get('best_val_acc', 0.0))
        except Exception:
            return 0.0

    rows.sort(key=sort_key, reverse=True)
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_md.write_text('No runs found.\n', encoding='utf-8')
        return

    cols = [
        'run_name',
        'model_name',
        'train_mode',
        'best_val_acc',
        'test_acc',
        'avg_epoch_time_sec',
        'trainable_params',
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    lines = []
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(c, '')) for c in cols) + ' |')

    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize outputs/*/metrics.json into a comparison table')
    parser.add_argument('--outputs_dir', type=str, default='outputs')
    parser.add_argument('--out_csv', type=str, default='outputs/summary.csv')
    parser.add_argument('--out_md', type=str, default='outputs/summary.md')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    rows = load_runs(outputs_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    print(f'Found {len(rows)} runs')
    print(f'Wrote: {out_csv}')
    print(f'Wrote: {out_md}')


if __name__ == '__main__':
    main()
