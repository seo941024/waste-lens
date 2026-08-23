"""ResNet18 학습 스크립트.

Baseline (backbone freeze, 증강 없음):
    python -m src.train --name baseline --no-augment --epochs 10

Improved (layer4 unfreeze + 증강):
    python -m src.train --name improved --unfreeze-layer4 --epochs 15 --lr 1e-4
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from configs.config import (BATCH_SIZE, CKPT_DIR, EPOCHS, LR, NUM_WORKERS,
                            RESULTS_DIR, SEED, WEIGHT_DECAY)
from src.dataset import build_dataloaders
from src.model import build_model, get_device, save_checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None, amp=False):
    train = optimizer is not None
    model.train() if train else model.eval()

    # Ampere 이상에서는 bf16을 쓴다. bf16은 fp16과 달리 지수부가 fp32와 같아
    # 값 범위가 넓으므로 GradScaler 없이 그대로 학습해도 안정적이다.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with autocast:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline", help="실험 이름 (체크포인트/로그 파일명)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="데이터 로딩 프로세스 수. GPU 학습이면 4~8 권장")
    parser.add_argument("--no-amp", action="store_true",
                        help="bf16 혼합정밀을 끄고 fp32로 학습")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    amp = (not args.no_amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 입력 크기가 고정이라 최적 커널을 캐싱
        print(f"device: {torch.cuda.get_device_name(0)} | bf16 AMP: {'ON' if amp else 'OFF'}")
    else:
        print("device: cpu  (GPU를 못 찾았습니다)")

    train_loader, val_loader = build_dataloaders(
        augment=not args.no_augment, batch_size=args.batch_size, num_workers=args.workers)
    print(f"train {len(train_loader.dataset)}장 / val {len(val_loader.dataset)}장 "
          f"| batch {args.batch_size} | workers {args.workers}")

    model = build_model(freeze_backbone=True, unfreeze_layer4=args.unfreeze_layer4)
    model = model.to(device, memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = CKPT_DIR / f"{args.name}.pt"
    history, best_acc = [], 0.0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, amp=amp)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckpt_path, {"val_acc": val_acc, "epoch": epoch})
            marker = "  <- best"

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{time.time() - started:.1f}s{marker}")

    log_path = RESULTS_DIR / f"{args.name}_history.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nbest val acc: {best_acc:.4f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"history: {log_path}")


if __name__ == "__main__":
    main()
