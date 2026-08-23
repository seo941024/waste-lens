"""Test셋 평가: Accuracy / Precision / Recall / F1 / Confusion Matrix,
오분류 사례, Threshold별 Coverage & Accepted Accuracy (계획서 6.3, 7장).

사용:
    python -m src.evaluate --name baseline
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from configs.classes import CLASSES
from configs.config import CKPT_DIR, RESULTS_DIR
from src.dataset import build_test_loader
from src.model import get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_probs, all_labels = [], []
    for images, labels in loader:
        probs = F.softmax(model(images.to(device)), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def threshold_analysis(probs, labels, thresholds):
    """임계값별 Coverage(답변 제공 비율)와 Accepted Accuracy(제공한 답의 정확도)."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    rows = []
    for t in thresholds:
        accepted = conf >= t
        coverage = float(accepted.mean())
        acc = float((preds[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0
        rows.append({"threshold": round(t, 2), "coverage": round(coverage, 4),
                     "accepted_accuracy": round(acc, 4), "n_accepted": int(accepted.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--topk-wrong", type=int, default=30,
                        help="High-confidence wrong prediction 상위 N건 저장")
    args = parser.parse_args()

    device = get_device()
    model, payload = load_checkpoint(CKPT_DIR / f"{args.name}.pt", device)
    loader = build_test_loader()
    print(f"test {len(loader.dataset)}장 | 체크포인트 val_acc={payload.get('val_acc')}")

    probs, labels = collect_predictions(model, loader, device)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    accuracy = float((preds == labels).mean())

    report = classification_report(labels, preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    plot_confusion_matrix(cm, RESULTS_DIR / f"{args.name}_confusion_matrix.png")

    thresholds = threshold_analysis(probs, labels, np.arange(0.0, 1.0, 0.05))
    print("\nthreshold  coverage  accepted_acc")
    for row in thresholds:
        print(f"   {row['threshold']:.2f}      {row['coverage']:.3f}      {row['accepted_accuracy']:.3f}")

    # High-confidence wrong prediction: 확신했는데 틀린 사례 (계획서 6.3)
    wrong = np.where(preds != labels)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:args.topk_wrong]
    samples = loader.dataset.samples
    hcw = [{"path": str(samples[i][0]), "true": CLASSES[labels[i]],
            "pred": CLASSES[preds[i]], "confidence": round(float(conf[i]), 4)}
           for i in wrong]

    out = {
        "name": args.name,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "threshold_analysis": thresholds,
        "high_confidence_wrong": hcw,
    }
    out_path = RESULTS_DIR / f"{args.name}_eval.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
