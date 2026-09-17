# ============================================================
# Evaluasi SSDLite MobileNetV3 — Deteksi Bahan Makanan
# Menghitung mAP, per-class AP, Precision, Recall, F1
# ============================================================

import os
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import csv
from collections import defaultdict

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR    = r"D:\Kuliah\Semester 8\Tugas Akhir (TA)\Skripsi New Model\EfficientDet"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_SSD")
MODEL_PATH  = os.path.join(BASE_DIR, "Output_SSD_3Fase", "fase3", "best_final.pt")
OUT_EVAL    = os.path.join(BASE_DIR, "Output_SSD_3Fase", "evaluasi")

TEST_IMG    = os.path.join(DATASET_DIR, "test")
TEST_JSON   = os.path.join(DATASET_DIR, "test", "_annotations.coco.json")

NUM_CLASSES = 11
IMG_SIZE    = 320
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.3
IOU_THRESH  = 0.5

CLASS_NAMES = {
    1:  "Bayam",
    2:  "Brokoli",
    3:  "Daging Ayam",
    4:  "Daging Sapi",
    5:  "Kentang",
    6:  "Tahu",
    7:  "Telur",
    8:  "Tempe",
    9:  "Tomat",
    10: "Wortel",
}

# ============================================================
# DATASET
# ============================================================
class CocoDataset(Dataset):
    def __init__(self, img_dir, json_path, img_size=320):
        self.img_dir  = img_dir
        self.img_size = img_size
        self.coco     = COCO(json_path)
        self.img_ids  = list(self.coco.imgs.keys())

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img            = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_tensor     = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1 = (x / orig_w) * self.img_size
            y1 = (y / orig_h) * self.img_size
            x2 = ((x + w) / orig_w) * self.img_size
            y2 = ((y + h) / orig_h) * self.img_size
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return img_tensor, {'boxes': boxes, 'labels': labels}, img_info['file_name']


def collate_fn(batch):
    imgs    = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    names   = [b[2] for b in batch]
    return imgs, targets, names


# ============================================================
# BUILD MODEL
# ============================================================
def build_model(num_classes, pretrained=True):
    if pretrained:
        model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
        print("✅ Pretrained COCO (91 kelas) loaded")

        in_channels = det_utils.retrieve_out_channels(
            model.backbone, (IMG_SIZE, IMG_SIZE)
        )
        num_anchors = model.anchor_generator.num_anchors_per_location()

        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=torch.nn.BatchNorm2d
        )
        print(f"✅ Classification head: 91 → {num_classes} kelas")
    else:
        model = ssdlite320_mobilenet_v3_large(
            weights=None, num_classes=num_classes
        )
        print("⚠️  Model tanpa pretrained weights")
    return model


# ============================================================
# IoU
# ============================================================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ============================================================
# AVERAGE PRECISION
# ============================================================
def compute_ap(recalls, precisions):
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return ap / 11.0


# ============================================================
# EVALUASI UTAMA
# ============================================================
def evaluate(model, loader, device, conf_thresh, iou_thresh):
    model.eval()
    all_preds    = defaultdict(list)
    all_gt_count = defaultdict(int)

    with torch.no_grad():
        for imgs, targets, _ in tqdm(loader, desc="Evaluasi"):
            imgs  = [img.to(device) for img in imgs]
            preds = model(imgs)

            for pred, target in zip(preds, targets):
                gt_boxes  = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()
                gt_used   = [False] * len(gt_boxes)

                for lbl in gt_labels:
                    all_gt_count[int(lbl)] += 1

                pred_boxes  = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                sorted_idx  = np.argsort(-pred_scores)

                for idx in sorted_idx:
                    score = pred_scores[idx]
                    if score < conf_thresh:
                        continue

                    pred_box   = pred_boxes[idx]
                    pred_label = int(pred_labels[idx])
                    best_iou   = 0.0
                    best_gt    = -1

                    for gt_idx, (gt_box, gt_label) in enumerate(
                            zip(gt_boxes, gt_labels)):
                        if int(gt_label) != pred_label:
                            continue
                        if gt_used[gt_idx]:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt  = gt_idx

                    if best_iou >= iou_thresh and best_gt >= 0:
                        all_preds[pred_label].append((score, 1))
                        gt_used[best_gt] = True
                    else:
                        all_preds[pred_label].append((score, 0))

    return all_preds, all_gt_count


# ============================================================
# HITUNG METRIK
# ============================================================
def compute_metrics(all_preds, all_gt_count, class_names):
    results = {}

    for cls_id, preds in all_preds.items():
        preds.sort(key=lambda x: -x[0])

        tp_cumsum  = 0
        fp_cumsum  = 0
        precisions = []
        recalls    = []
        n_gt       = all_gt_count.get(cls_id, 0)

        for score, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec  = tp_cumsum / n_gt if n_gt > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        ap   = compute_ap(recalls, precisions)
        tp   = tp_cumsum
        fp   = fp_cumsum
        fn   = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[cls_id] = {
            'name'     : class_names.get(cls_id, f"Class {cls_id}"),
            'AP'       : ap,
            'Precision': prec,
            'Recall'   : rec,
            'F1'       : f1,
            'TP'       : tp,
            'FP'       : fp,
            'FN'       : fn,
            'n_gt'     : n_gt,
        }

    mAP = np.mean([v['AP'] for v in results.values()]) if results else 0.0
    return results, mAP


# ============================================================
# VISUALISASI PREDIKSI
# ============================================================
def visualize_predictions(model, dataset, device, out_dir,
                           conf_thresh=0.3, n_samples=20):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std= [1/0.229,       1/0.224,      1/0.225]
    )

    indices = np.random.choice(
        len(dataset), min(n_samples, len(dataset)), replace=False
    )

    for i, idx in enumerate(indices):
        img_tensor, target, fname = dataset[idx]

        with torch.no_grad():
            pred = model([img_tensor.to(device)])[0]

        img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        img_show = np.clip(img_show, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(fname, fontsize=10)

        axes[0].imshow(img_show)
        axes[0].set_title("Ground Truth", color='green')
        for box, lbl in zip(target['boxes'].numpy(),
                            target['labels'].numpy()):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-4,
                         CLASS_NAMES.get(int(lbl), str(lbl)),
                         color='lime', fontsize=8,
                         bbox=dict(facecolor='black', alpha=0.5, pad=1))

        axes[1].imshow(img_show)
        axes[1].set_title("Prediksi Model", color='red')
        for box, lbl, score in zip(
                pred['boxes'].cpu().numpy(),
                pred['labels'].cpu().numpy(),
                pred['scores'].cpu().numpy()):
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x1, y1-4,
                f"{CLASS_NAMES.get(int(lbl), str(lbl))} {score:.2f}",
                color='red', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"sample_{i+1:03d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"  📸 {len(indices)} contoh prediksi disimpan di: {out_dir}")


# ============================================================
# PLOT METRIK
# ============================================================
def plot_eval_results(results, mAP, out_dir):
    cls_names = [v['name']      for v in results.values()]
    ap_values = [v['AP']        for v in results.values()]
    prec_vals = [v['Precision'] for v in results.values()]
    rec_vals  = [v['Recall']    for v in results.values()]
    f1_vals   = [v['F1']        for v in results.values()]

    x     = np.arange(len(cls_names))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    bars = axes[0].bar(x, ap_values, color='steelblue', alpha=0.8)
    axes[0].axhline(mAP, color='red', linestyle='--',
                    label=f'mAP = {mAP:.4f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cls_names, rotation=30, ha='right')
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('AP per Kelas (IoU=0.5)')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for bar, val in zip(bars, ap_values):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9
        )

    axes[1].bar(x - width, prec_vals, width,
                label='Precision', color='steelblue', alpha=0.8)
    axes[1].bar(x,          rec_vals, width,
                label='Recall',    color='coral',     alpha=0.8)
    axes[1].bar(x + width, f1_vals,  width,
                label='F1-Score',  color='green',     alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cls_names, rotation=30, ha='right')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision / Recall / F1 per Kelas')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "eval_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Grafik evaluasi disimpan: {path}")

# ============================================================
# CARI CONTOH KESALAHAN SPESIFIK PER KELAS
# ============================================================

def visualize_kesalahan_per_kelas(model, dataset, device, out_dir,
                                   target_classes, class_names,
                                   conf_thresh=0.3, iou_thresh=0.5,
                                   n_samples=5):
    """
    Mencari gambar di mana kelas target mengalami kesalahan:
    - FN: GT ada tapi tidak terdeteksi dengan benar
    - FP/Misclass: prediksi kelas target tapi salah (IoU<thresh atau kelas beda)
    """
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std= [1/0.229,       1/0.224,      1/0.225]
    )

    for cls_id, cls_name in target_classes.items():
        cls_dir = os.path.join(out_dir, f"kesalahan_{cls_name.replace(' ', '_')}")
        os.makedirs(cls_dir, exist_ok=True)

        found = []

        for idx in range(len(dataset)):
            if len(found) >= n_samples:
                break

            img_tensor, target, fname = dataset[idx]
            gt_boxes  = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()

            if cls_id not in gt_labels:
                continue

            with torch.no_grad():
                pred = model([img_tensor.to(device)])[0]

            pred_boxes  = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            gt_used = [False] * len(gt_boxes)
            sorted_idx = np.argsort(-pred_scores)

            has_error = False

            for i in sorted_idx:
                score = pred_scores[i]
                if score < conf_thresh:
                    continue
                pred_box   = pred_boxes[i]
                pred_label = int(pred_labels[i])

                best_iou, best_gt = 0.0, -1
                for gt_idx, (gt_box, gt_label) in enumerate(
                        zip(gt_boxes, gt_labels)):
                    if gt_used[gt_idx]:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_gt = iou, gt_idx

                if best_iou >= iou_thresh and best_gt >= 0:
                    gt_used[best_gt] = True
                    # Salah kelas meski IoU cocok
                    if int(gt_labels[best_gt]) == cls_id and pred_label != cls_id:
                        has_error = True
                    if pred_label == cls_id and int(gt_labels[best_gt]) != cls_id:
                        has_error = True
                else:
                    # FP murni kelas target
                    if pred_label == cls_id:
                        has_error = True

            # FN: GT kelas target tidak tertandingi
            for gt_idx, (gt_label, used) in enumerate(zip(gt_labels, gt_used)):
                if int(gt_label) == cls_id and not used:
                    has_error = True

            if has_error:
                found.append((idx, img_tensor, target, fname, pred))

        print(f"\n📸 Kelas '{cls_name}': ditemukan {len(found)} gambar bermasalah")

        for i, (idx, img_tensor, target, fname, pred) in enumerate(found):
            img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
            img_show = np.clip(img_show, 0, 1)

            gt_boxes  = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            pred_boxes  = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Kelas: {cls_name} (kasus bermasalah) | {fname}",
                         fontsize=11, fontweight='bold')

            axes[0].imshow(img_show)
            axes[0].set_title("Ground Truth", color='green', fontweight='bold')
            for box, lbl in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                color = 'lime' if int(lbl) == cls_id else 'cyan'
                rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                axes[0].add_patch(rect)
                axes[0].text(x1, max(y1-5,0), class_names.get(int(lbl), str(lbl)),
                             color=color, fontsize=9, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, pad=2))

            axes[1].imshow(img_show)
            axes[1].set_title("Prediksi Model", color='red', fontweight='bold')
            for box, lbl, score in zip(pred_boxes, pred_labels, pred_scores):
                if score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                color = 'red' if int(lbl) == cls_id else 'orange'
                rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)
                axes[1].text(x1, max(y1-5,0), f"{class_names.get(int(lbl), str(lbl))} {score:.2f}",
                             color=color, fontsize=9, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.6, pad=2))

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(cls_dir, f"{cls_name.replace(' ','_')}_error_{i+1:02d}.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

        print(f"  ✅ Disimpan di: {cls_dir}")


# ============================================================
# VISUALISASI KESALAHAN DETEKSI
# ============================================================
def visualize_errors(model, dataset, device, out_dir,
                     conf_thresh=0.3, iou_thresh=0.5,
                     max_per_type=3):
    """
    Menyimpan contoh gambar untuk tiga pola kesalahan:
    - FN: objek ada tapi tidak terdeteksi (fokus: Bayam)
    - FP: deteksi muncul di area background (fokus: Daging Ayam, Tempe)
    - Konfusi: kelas tertukar (fokus: Daging Ayam <-> Daging Sapi)
    """
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std= [1/0.229,       1/0.224,      1/0.225]
    )

    # Target kelas untuk setiap pola kesalahan
    FN_TARGET    = {1: "Bayam"}                    # kelas FN tinggi
    FP_TARGET    = {3: "Daging Ayam", 8: "Tempe"}  # kelas FP tinggi
    CONF_TARGET  = {                               # pasang konfusi
        (3, 4): ("Daging Ayam", "Daging Sapi"),
        (4, 3): ("Daging Sapi", "Daging Ayam"),
    }

    fn_cases   = {cls_id: [] for cls_id in FN_TARGET}
    fp_cases   = {cls_id: [] for cls_id in FP_TARGET}
    conf_cases = {pair: []   for pair   in CONF_TARGET}

    with torch.no_grad():
        for idx in range(len(dataset)):
            if all(
                len(fn_cases[c])   >= max_per_type for c in fn_cases
            ) and all(
                len(fp_cases[c])   >= max_per_type for c in fp_cases
            ) and all(
                len(conf_cases[p]) >= max_per_type for p in conf_cases
            ):
                break

            img_tensor, target, fname = dataset[idx]
            pred = model([img_tensor.to(device)])[0]

            img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
            img_show = np.clip(img_show, 0, 1)

            gt_boxes  = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            gt_used   = [False] * len(gt_boxes)

            pred_boxes  = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            sorted_idx  = np.argsort(-pred_scores)

            matched_preds = []

            for i in sorted_idx:
                score = pred_scores[i]
                if score < conf_thresh:
                    continue

                pred_box   = pred_boxes[i]
                pred_label = int(pred_labels[i])
                best_iou, best_gt, best_gt_label = 0.0, -1, -1

                for gt_idx, (gt_box, gt_label) in enumerate(
                        zip(gt_boxes, gt_labels)):
                    if gt_used[gt_idx]:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou      = iou
                        best_gt       = gt_idx
                        best_gt_label = int(gt_label)

                if best_iou >= iou_thresh and best_gt >= 0:
                    gt_used[best_gt] = True
                    matched_preds.append((pred_box, pred_label,
                                          score, best_gt_label, 'tp'))

                    # Cek konfusi antar kelas
                    pair = (best_gt_label, pred_label)
                    if pair in conf_cases and \
                       len(conf_cases[pair]) < max_per_type and \
                       best_gt_label != pred_label:
                        conf_cases[pair].append(
                            (img_show, gt_boxes, gt_labels,
                             pred_boxes, pred_labels, pred_scores,
                             fname, pred_box, best_gt_label, pred_label)
                        )
                else:
                    matched_preds.append((pred_box, pred_label,
                                          score, -1, 'fp'))

                    # Cek FP untuk kelas target
                    if pred_label in fp_cases and \
                       len(fp_cases[pred_label]) < max_per_type:
                        fp_cases[pred_label].append(
                            (img_show, gt_boxes, gt_labels,
                             pred_boxes, pred_labels, pred_scores,
                             fname, pred_box)
                        )

            # Cek FN untuk kelas target
            for gt_idx, (gt_label, used) in enumerate(
                    zip(gt_labels, gt_used)):
                cls_id = int(gt_label)
                if not used and cls_id in fn_cases and \
                   len(fn_cases[cls_id]) < max_per_type:
                    fn_cases[cls_id].append(
                        (img_show, gt_boxes, gt_labels,
                         pred_boxes, pred_labels, pred_scores,
                         fname, gt_boxes[gt_idx])
                    )

    # ── Simpan gambar FN ──────────────────────────────────
    fn_dir = os.path.join(out_dir, "error_fn")
    os.makedirs(fn_dir, exist_ok=True)

    for cls_id, cases in fn_cases.items():
        cls_name = FN_TARGET[cls_id]
        for i, (img_show, gt_boxes, gt_labels, pred_boxes,
                pred_labels, pred_scores, fname, missed_box) in \
                enumerate(cases):

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"False Negative — {cls_name}\n{fname}",
                fontsize=11, fontweight='bold'
            )

            # Kiri: Ground Truth
            axes[0].imshow(img_show)
            axes[0].set_title("Ground Truth", color='green',
                               fontweight='bold')
            for box, lbl in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                color = 'lime' if int(lbl) == cls_id else 'cyan'
                lw    = 3      if int(lbl) == cls_id else 1
                rect  = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=lw, edgecolor=color, facecolor='none'
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x1, y1-4, CLASS_NAMES.get(int(lbl), str(lbl)),
                    color=color, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            # Kanan: Prediksi (objek yang terlewat ditandai)
            axes[1].imshow(img_show)
            axes[1].set_title("Prediksi Model (objek terlewat)",
                               color='red', fontweight='bold')
            for box, lbl, score in zip(pred_boxes, pred_labels,
                                        pred_scores):
                if score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[1].add_patch(rect)
                axes[1].text(
                    x1, y1-4,
                    f"{CLASS_NAMES.get(int(lbl), str(lbl))} "
                    f"{score:.2f}",
                    color='red', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )
            # Tandai objek yang terlewat dengan kotak kuning
            x1, y1, x2, y2 = missed_box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='yellow',
                facecolor='yellow', alpha=0.2,
                linestyle='--'
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x1, y2+4, f"❌ Terlewat: {cls_name}",
                color='yellow', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2)
            )

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(
                fn_dir, f"fn_{cls_name.lower().replace(' ','_')}"
                        f"_{i+1:02d}.png"
            )
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

    print(f"  ❌ FN cases disimpan di: {fn_dir}")

    # ── Simpan gambar FP ──────────────────────────────────
    fp_dir = os.path.join(out_dir, "error_fp")
    os.makedirs(fp_dir, exist_ok=True)

    for cls_id, cases in fp_cases.items():
        cls_name = FP_TARGET[cls_id]
        for i, (img_show, gt_boxes, gt_labels, pred_boxes,
                pred_labels, pred_scores, fname, fp_box) in \
                enumerate(cases):

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"False Positive — {cls_name}\n{fname}",
                fontsize=11, fontweight='bold'
            )

            axes[0].imshow(img_show)
            axes[0].set_title("Ground Truth", color='green',
                               fontweight='bold')
            for box, lbl in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x1, y1-4, CLASS_NAMES.get(int(lbl), str(lbl)),
                    color='lime', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            axes[1].imshow(img_show)
            axes[1].set_title("Prediksi Model (FP ditandai)",
                               color='red', fontweight='bold')
            for box, lbl, score in zip(pred_boxes, pred_labels,
                                        pred_scores):
                if score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                is_fp_box = np.allclose(box, fp_box, atol=1.0)
                color = 'orange' if is_fp_box else 'red'
                lw    = 3        if is_fp_box else 1
                rect  = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=lw, edgecolor=color, facecolor='none'
                )
                axes[1].add_patch(rect)
                label_txt = (
                    f"⚠️ FP: {CLASS_NAMES.get(int(lbl), str(lbl))} "
                    f"{score:.2f}"
                    if is_fp_box else
                    f"{CLASS_NAMES.get(int(lbl), str(lbl))} {score:.2f}"
                )
                axes[1].text(
                    x1, y1-4, label_txt,
                    color=color, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(
                fp_dir, f"fp_{cls_name.lower().replace(' ','_')}"
                        f"_{i+1:02d}.png"
            )
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

    print(f"  ⚠️  FP cases disimpan di: {fp_dir}")

    # ── Simpan gambar Konfusi ─────────────────────────────
    conf_dir = os.path.join(out_dir, "error_konfusi")
    os.makedirs(conf_dir, exist_ok=True)

    for pair, cases in conf_cases.items():
        gt_cls_name   = CONF_TARGET[pair][0]
        pred_cls_name = CONF_TARGET[pair][1]
        for i, (img_show, gt_boxes, gt_labels, pred_boxes,
                pred_labels, pred_scores, fname,
                conf_box, gt_lbl, pred_lbl) in enumerate(cases):

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"Konfusi: {gt_cls_name} → {pred_cls_name}\n{fname}",
                fontsize=11, fontweight='bold'
            )

            axes[0].imshow(img_show)
            axes[0].set_title("Ground Truth", color='green',
                               fontweight='bold')
            for box, lbl in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x1, y1-4, CLASS_NAMES.get(int(lbl), str(lbl)),
                    color='lime', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            axes[1].imshow(img_show)
            axes[1].set_title("Prediksi Model (konfusi ditandai)",
                               color='red', fontweight='bold')
            for box, lbl, score in zip(pred_boxes, pred_labels,
                                        pred_scores):
                if score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                is_conf = np.allclose(box, conf_box, atol=1.0)
                color   = 'orange' if is_conf else 'red'
                lw      = 3        if is_conf else 1
                rect    = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=lw, edgecolor=color, facecolor='none'
                )
                axes[1].add_patch(rect)
                label_txt = (
                    f"⚠️ {CLASS_NAMES.get(int(lbl), str(lbl))} "
                    f"{score:.2f}\n(GT: {gt_cls_name})"
                    if is_conf else
                    f"{CLASS_NAMES.get(int(lbl), str(lbl))} {score:.2f}"
                )
                axes[1].text(
                    x1, y1-4, label_txt,
                    color=color, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(
                conf_dir,
                f"konfusi_{gt_cls_name.lower().replace(' ','_')}"
                f"_vs_{pred_cls_name.lower().replace(' ','_')}"
                f"_{i+1:02d}.png"
            )
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

    print(f"  🔀 Konfusi cases disimpan di: {conf_dir}")

# ============================================================
# SIMPAN CSV & JSON
# ============================================================
def save_results(results, mAP, out_dir):
    csv_path = os.path.join(out_dir, "eval_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'AP', 'Precision', 'Recall',
                         'F1', 'TP', 'FP', 'FN', 'n_GT'])
        for v in results.values():
            writer.writerow([
                v['name'],
                f"{v['AP']:.4f}",
                f"{v['Precision']:.4f}",
                f"{v['Recall']:.4f}",
                f"{v['F1']:.4f}",
                v['TP'], v['FP'], v['FN'], v['n_gt']
            ])
        writer.writerow(['mAP', f"{mAP:.4f}", '', '', '', '', '', '', ''])
    print(f"  📄 CSV disimpan: {csv_path}")

    json_path = os.path.join(out_dir, "eval_results.json")
    output = {
        'mAP'    : round(mAP, 6),
        'classes': {
            v['name']: {
                'AP'       : round(v['AP'],        6),
                'Precision': round(v['Precision'], 6),
                'Recall'   : round(v['Recall'],    6),
                'F1'       : round(v['F1'],        6),
                'TP'       : v['TP'],
                'FP'       : v['FP'],
                'FN'       : v['FN'],
                'n_GT'     : v['n_gt'],
            } for v in results.values()
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  📄 JSON disimpan: {json_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    os.makedirs(OUT_EVAL, exist_ok=True)
    print(f"✅ Device : {DEVICE}")
    print(f"✅ GPU    : {torch.cuda.get_device_name(0)}")
    print(f"📦 Model  : {MODEL_PATH}")
    print(f"📁 Output : {OUT_EVAL}")

    # Dataset
    print("\n📂 Loading test dataset...")
    test_dataset = CocoDataset(TEST_IMG, TEST_JSON, IMG_SIZE)
    test_loader  = DataLoader(
        test_dataset, batch_size=8,
        shuffle=False, num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"✅ Test: {len(test_dataset):,} gambar ({len(test_loader)} batch)")

    # Load Model
    print("\n🔧 Loading model...")
    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    )
    model.eval()
    print("✅ Model loaded!")

    # Evaluasi
    print(f"\n🔍 Evaluasi dengan conf={CONF_THRESH}, IoU={IOU_THRESH}...")
    all_preds, all_gt_count = evaluate(
        model, test_loader, DEVICE, CONF_THRESH, IOU_THRESH
    )

    # Metrik
    results, mAP = compute_metrics(all_preds, all_gt_count, CLASS_NAMES)

    # Print hasil
    print(f"\n{'='*60}")
    print(f"  📊 HASIL EVALUASI (IoU={IOU_THRESH})")
    print(f"{'='*60}")
    print(f"  {'Kelas':<20} {'AP':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*60}")
    for v in results.values():
        print(f"  {v['name']:<20} "
              f"{v['AP']:>6.3f} "
              f"{v['Precision']:>6.3f} "
              f"{v['Recall']:>6.3f} "
              f"{v['F1']:>6.3f} "
              f"{v['TP']:>5} "
              f"{v['FP']:>5} "
              f"{v['FN']:>5}")
    print(f"  {'-'*60}")
    print(f"  {'mAP@0.5':<20} {mAP:.2f}")
    print(f"{'='*60}")

    # Simpan hasil
    save_results(results, mAP, OUT_EVAL)

    # Plot metrik
    plot_eval_results(results, mAP, OUT_EVAL)

# ============================================================
# PANGGIL DI MAIN — tambahkan setelah visualize_predictions()
# ============================================================

    # Kelas yang dianalisis di 5.2.3
    ANALISIS_CLASSES = {
        9 : "Tomat",
        10: "Wortel",
        3 : "Daging Ayam",
        1 : "Bayam",
    }

    print("\n📸 Mencari contoh kesalahan deteksi per kelas...")
    visualize_kesalahan_per_kelas(
        model=model, dataset=test_dataset, device=DEVICE,
        out_dir=os.path.join(OUT_EVAL, "kesalahan_per_kelas"),
        target_classes=ANALISIS_CLASSES,
        class_names=CLASS_NAMES,
        conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH,
        n_samples=5
    )


    # Visualisasi Kesalahan
    print("\n🔍 Membuat visualisasi kesalahan deteksi...")
    visualize_errors(
        model, test_dataset, DEVICE,
        out_dir      = OUT_EVAL,
        conf_thresh  = CONF_THRESH,
        iou_thresh   = IOU_THRESH,
        max_per_type = 3
    )

    # Visualisasi prediksi
    print("\n📸 Membuat visualisasi prediksi...")
    visualize_predictions(
        model, test_dataset, DEVICE,
        out_dir     = os.path.join(OUT_EVAL, "visualisasi"),
        conf_thresh = CONF_THRESH,
        n_samples   = 20
    )

    print(f"\n🎉 Evaluasi selesai! Semua hasil di: {OUT_EVAL}")