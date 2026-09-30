"""
tools/benchmark_shape_cnn.py
=============================
Phase-1 diagnostic benchmark for mycv.shape_cnn: measures actual accuracy
across clean / rotated / noisy / degraded synthetic distributions, with
confusion matrices and per-class precision/recall/F1 -- rather than
trusting a single "100% validation accuracy" number that only describes
the clean synthetic distribution the model was trained on.

Usage:
    python3 tools/benchmark_shape_cnn.py
    python3 tools/benchmark_shape_cnn.py --weights path/to/other.npz

Sections
--------
A. Clean synthetic          -- confusion matrix + per-class P/R/F1
B. Per-degradation accuracy  -- one accuracy number per degradation type
C. Rotation-binned accuracy  -- accuracy at each of 0/15/30/45/60/75/90 deg,
                                 for Square and Rectangle separately
D. Circle-specific stress    -- accuracy vs radius, and vs each degradation,
                                 isolated to the Circle class
E. Combined (rotation + degradation + real-mask-like) held-out set
"""
import argparse
import sys
sys.path.insert(0, "/home/claude")

import numpy as np

from mycv.geometry import rotate_image
from mycv.shape_cnn import CLASS_NAMES, load_shape_cnn, predict_shape, _resize_mask_to_input
from tools.degrade import DEGRADATIONS


# ---------------------------------------------------------------------------
# Shape renderers (shared with gen_shape_dataset.py's approach: draw on a
# generous canvas, crop to bbox happens inside predict_shape/_resize_mask_to_input)
# ---------------------------------------------------------------------------

def render_circle(rng, radius=None, base=100):
    r = radius if radius is not None else rng.uniform(10.0, 35.0)
    cx = base / 2 + rng.uniform(-3, 3)
    cy = base / 2 + rng.uniform(-3, 3)
    yy, xx = np.mgrid[0:base, 0:base]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.float64)


def render_rect(rng, angle=None, force_square=True, base=100):
    if force_square:
        w = h = rng.uniform(20.0, 50.0)
    else:
        w = rng.uniform(18.0, 55.0)
        ratio = rng.choice([rng.uniform(1.4, 2.2), rng.uniform(1 / 2.2, 1 / 1.4)])
        h = np.clip(w / ratio, 10.0, 60.0)

    canvas = np.zeros((base, base), dtype=np.float64)
    cy = cx = base / 2
    y0, y1 = int(cy - h / 2), int(cy + h / 2)
    x0, x1 = int(cx - w / 2), int(cx + w / 2)
    canvas[y0:y1, x0:x1] = 255.0

    if angle is None:
        angle = rng.uniform(0, 45) if force_square else rng.uniform(-25, 25)
    rotated = rotate_image(canvas, angle)
    return (rotated >= 127).astype(np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names=CLASS_NAMES, title=""):
    if title:
        print(f"\n{title}")
    header = "Actual\\Pred".ljust(12) + "".join(n[:6].rjust(8) for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = name[:10].ljust(12) + "".join(str(cm[i, j]).rjust(8) for j in range(len(class_names)))
        print(row)
    acc = np.trace(cm) / max(1, cm.sum())
    print(f"overall accuracy: {acc:.4f}  (n={cm.sum()})")
    return acc


def precision_recall_f1(cm, class_names=CLASS_NAMES):
    print(f"{'class':12s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{name:12s} {precision:10.3f} {recall:10.3f} {f1:10.3f} {support:8d}")


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------

def gen_clean_set(n_per_class, seed):
    rng = np.random.default_rng(seed)
    masks, labels = [], []
    for _ in range(n_per_class):
        masks.append(render_circle(rng)); labels.append(0)
    for _ in range(n_per_class):
        masks.append(render_rect(rng, force_square=True)); labels.append(1)
    for _ in range(n_per_class):
        masks.append(render_rect(rng, force_square=False)); labels.append(2)
    return masks, labels


def section_a_clean(model, n_per_class=150, seed=1000):
    masks, labels = gen_clean_set(n_per_class, seed)
    preds = [np.argmax(list(predict_shape(m, model=model)["probs"].values())) for m in masks]
    cm = confusion_matrix(labels, preds)
    acc = print_confusion_matrix(cm, title="=== A. Clean synthetic (held-out seed) ===")
    precision_recall_f1(cm)
    return acc


def section_b_degradations(model, n_per_class=80, seed=2000):
    print("\n=== B. Per-degradation accuracy (held-out shapes, seed={}) ===".format(seed))
    print(f"{'degradation':18s} {'accuracy':>10s}")
    results = {}
    for deg_name, deg_fn in DEGRADATIONS.items():
        rng = np.random.default_rng(seed)
        masks, labels = gen_clean_set(n_per_class, seed)
        degraded = [deg_fn(m, rng) for m in masks]
        preds = [np.argmax(list(predict_shape(m, model=model)["probs"].values())) for m in degraded]
        acc = np.mean(np.array(preds) == np.array(labels))
        results[deg_name] = acc
        print(f"{deg_name:18s} {acc:10.4f}")
    return results


def section_c_rotation_binned(model, n_per_bin=40, seed=3000):
    print("\n=== C. Rotation-binned accuracy (Square and Rectangle) ===")
    print("(angle passed directly to the renderer -- no remapping)")
    for force_square, class_idx, label, angles in [
        (True, 1, "Square", [0, 15, 30, 45, 60, 75, 90]),
        (False, 2, "Rectangle", [-90, -60, -30, 0, 30, 60, 90]),
    ]:
        print(f"\n{label}:")
        print(f"{'angle':>8s} {'accuracy':>10s}")
        for angle in angles:
            rng = np.random.default_rng(seed + int(angle) + 1000)
            correct = 0
            for _ in range(n_per_bin):
                m = render_rect(rng, angle=angle, force_square=force_square)
                r = predict_shape(m, model=model)
                pred_idx = CLASS_NAMES.index(r["label"])
                correct += int(pred_idx == class_idx)
            acc = correct / n_per_bin
            print(f"{angle:8d} {acc:10.4f}")


def section_d_circle_stress(model, n_per_bin=40, seed=4000):
    print("\n=== D. Circle-specific stress test ===")
    print("\nBy radius:")
    print(f"{'radius':>8s} {'accuracy':>10s}")
    for radius in [6, 10, 15, 20, 25, 30, 35]:
        rng = np.random.default_rng(seed + radius)
        correct = 0
        for _ in range(n_per_bin):
            m = render_circle(rng, radius=radius)
            r = predict_shape(m, model=model)
            correct += int(r["label"] == "Circle")
        print(f"{radius:8d} {correct / n_per_bin:10.4f}")

    print("\nBy degradation (circles only):")
    print(f"{'degradation':18s} {'accuracy':>10s}")
    for deg_name, deg_fn in DEGRADATIONS.items():
        rng = np.random.default_rng(seed)
        correct = 0
        for _ in range(n_per_bin):
            m = render_circle(rng)
            m = deg_fn(m, rng)
            r = predict_shape(m, model=model)
            correct += int(r["label"] == "Circle")
        print(f"{deg_name:18s} {correct / n_per_bin:10.4f}")


def section_e_combined_stress(model, n_per_class=100, seed=5000):
    """
    The single hardest, most realistic test: rotation AND degradation AND
    small size, combined, per the plan's item 11 (balance difficulty, not
    just class counts).
    """
    print("\n=== E. Combined stress test (rotation + degradation + small size) ===")
    rng = np.random.default_rng(seed)
    deg_names = list(DEGRADATIONS.keys())
    masks, labels = [], []
    for _ in range(n_per_class):
        m = render_circle(rng, radius=rng.uniform(6, 20))
        m = DEGRADATIONS[rng.choice(deg_names)](m, rng)
        masks.append(m); labels.append(0)
    for _ in range(n_per_class):
        m = render_rect(rng, angle=rng.uniform(0, 45), force_square=True)
        m = DEGRADATIONS[rng.choice(deg_names)](m, rng)
        masks.append(m); labels.append(1)
    for _ in range(n_per_class):
        m = render_rect(rng, angle=rng.uniform(-25, 25), force_square=False)
        m = DEGRADATIONS[rng.choice(deg_names)](m, rng)
        masks.append(m); labels.append(2)

    preds = [np.argmax(list(predict_shape(m, model=model)["probs"].values())) for m in masks]
    cm = confusion_matrix(labels, preds)
    acc = print_confusion_matrix(cm, title="Combined stress confusion matrix:")
    precision_recall_f1(cm)
    return acc


def section_f_real_masks(model, real_dir):
    """
    Evaluate against REAL captured masks (from tools/capture_shape_masks.py),
    if any exist at `real_dir`. This is the actual test set the whole
    benchmark exists to eventually be checked against -- everything else
    in this file is synthetic. Skipped automatically if `real_dir` doesn't
    exist or contains no captures yet.
    """
    from pathlib import Path
    root = Path(real_dir)
    class_dirs = {"circle": 0, "square": 1, "rectangle": 2}

    masks, labels = [], []
    for name, idx in class_dirs.items():
        class_dir = root / name
        if not class_dir.exists():
            continue
        for npy_path in sorted(class_dir.glob("*.npy")):
            masks.append(np.load(npy_path))
            labels.append(idx)

    if not masks:
        print(f"\n=== F. Real captured masks ===")
        print(f"No real masks found at '{real_dir}'. Run tools/capture_shape_masks.py "
              f"to collect some -- everything above this line is synthetic-only.")
        return None

    preds = []
    for m in masks:
        try:
            r = predict_shape(m, model=model)
            preds.append(CLASS_NAMES.index(r["label"]))
        except ValueError:
            preds.append(-1)  # empty/invalid capture, counts as a miss

    valid = [(t, p) for t, p in zip(labels, preds) if p >= 0]
    cm = confusion_matrix([t for t, p in valid], [p for t, p in valid])
    acc = print_confusion_matrix(cm, title=f"=== F. Real captured masks (n={len(masks)}, from '{real_dir}') ===")
    precision_recall_f1(cm)
    n_invalid = len(masks) - len(valid)
    if n_invalid:
        print(f"({n_invalid} capture(s) were empty/invalid and excluded)")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--real-dir", type=str, default="dataset/real",
                         help="Directory of real masks from tools/capture_shape_masks.py")
    args = parser.parse_args()

    model = load_shape_cnn(args.weights)

    print("#" * 70)
    print("mycv.shape_cnn benchmark")
    print("#" * 70)

    acc_a = section_a_clean(model)
    results_b = section_b_degradations(model)
    section_c_rotation_binned(model)
    section_d_circle_stress(model)
    acc_e = section_e_combined_stress(model)
    acc_f = section_f_real_masks(model, args.real_dir)

    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    print(f"Clean synthetic accuracy:    {acc_a:.4f}")
    print(f"Combined stress accuracy:    {acc_e:.4f}")
    if acc_f is not None:
        print(f"REAL captured mask accuracy: {acc_f:.4f}  <-- this is the number that actually matters")
    else:
        print(f"REAL captured mask accuracy: (none collected yet -- run tools/capture_shape_masks.py)")
    print(f"Worst single degradation:    {min(results_b, key=results_b.get)} ({min(results_b.values()):.4f})")
    print(f"Best single degradation:     {max(results_b, key=results_b.get)} ({max(results_b.values()):.4f})")


if __name__ == "__main__":
    main()
