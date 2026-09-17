# ============================================================
# SSDLite MobileNetV3 Training — Deteksi Bahan Makanan
# Versi Improved: 3 Fase + Differential LR + Gradient Accumulation
# Windows + RTX 3050 4GB
# ============================================================

import os
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR    = r"D:\Kuliah\Semester 8\Tugas Akhir (TA)\Skripsi New Model\EfficientDet"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_SSD")

TRAIN_IMG   = os.path.join(DATASET_DIR, "train")
TRAIN_JSON  = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
VALID_IMG   = os.path.join(DATASET_DIR, "valid")
VALID_JSON  = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")

NUM_CLASSES = 11        # 10 kelas + 1 background
IMG_SIZE    = 320
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epoch per fase — dioptimalkan untuk hasil terbaik
EPOCHS_FASE1 = 30       # warm-up: cukup 30 untuk head beradaptasi
EPOCHS_FASE2 = 100      # fine-tuning utama: dinaikkan dari 50 → 100
EPOCHS_FASE3 = 50       # polishing: squeeze performa terakhir

ACCUM_STEPS  = 4        # gradient accumulation → batch efektif = 4×8 = 32

print(f"✅ Device : {DEVICE}")
print(f"✅ GPU    : {torch.cuda.get_device_name(0)}")
print(f"📋 Konfigurasi epoch: Fase1={EPOCHS_FASE1} | "
      f"Fase2={EPOCHS_FASE2} | Fase3={EPOCHS_FASE3}")
print(f"📋 Batch efektif    : {ACCUM_STEPS} × 8 = {ACCUM_STEPS * 8}")


# ============================================================
# DATASET CLASS
# ============================================================
class CocoDataset(Dataset):
    def __init__(self, img_dir, json_path, img_size=320, use_cache=False):
        self.img_dir   = img_dir
        self.img_size  = img_size
        self.coco      = COCO(json_path)
        self.img_ids   = list(self.coco.imgs.keys())
        self.use_cache = use_cache
        self.cache     = {}

        # Augmentasi sudah dilakukan di Roboflow
        # Di sini hanya normalisasi standar
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

        # Gunakan cache jika tersedia (hemat disk I/O)
        if self.use_cache and idx in self.cache:
            img = self.cache[idx]
        else:
            img = Image.open(img_path).convert("RGB")
            if self.use_cache:
                self.cache[idx] = img

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

        return img_tensor, {'boxes': boxes, 'labels': labels}


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


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
# DIFFERENTIAL LEARNING RATE
# ============================================================
def get_optimizer_diff_lr(model, backbone_lr, head_lr, weight_decay=1e-4):
    """
    Backbone: LR kecil  → jaga fitur pretrained yang sudah baik
    Head    : LR besar  → belajar kelas baru lebih cepat
    """
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"   Backbone params : {n_bb:,} (LR: {backbone_lr})")
    print(f"   Head params     : {n_hd:,} (LR: {head_lr})")

    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params,     'lr': head_lr},
    ], weight_decay=weight_decay)


# ============================================================
# TRAINING 1 EPOCH (dengan Gradient Accumulation)
# ============================================================
def train_one_epoch(model, loader, optimizer, device, epoch,
                    accum_steps=4):
    """
    Gradient Accumulation: update bobot setiap accum_steps batch
    Batch efektif = accum_steps × batch_size = 4 × 8 = 32
    Hasil: training lebih stabil, terutama untuk kelas minoritas
    """
    model.train()
    total_loss = 0
    count      = 0
    loop       = tqdm(loader, desc=f"  Train E{epoch}")

    optimizer.zero_grad()

    for batch_idx, (imgs, targets) in enumerate(loop):
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(t['boxes'].shape[0] == 0 for t in targets):
            continue

        loss_dict = model(imgs, targets)
        loss      = sum(l for l in loss_dict.values())
        loss      = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=3.0
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        count      += 1
        loop.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    # Flush sisa gradient
    if count % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / count if count > 0 else 0.0


# ============================================================
# VALIDASI
# ============================================================
def validate(model, loader, device, epoch):
    model.train()   # SSD hitung loss hanya di mode train
    total_loss = 0
    count      = 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"  Valid E{epoch}")
        for imgs, targets in loop:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if all(t['boxes'].shape[0] == 0 for t in targets):
                continue

            loss_dict  = model(imgs, targets)
            loss       = sum(l for l in loss_dict.values())
            total_loss += loss.item()
            count      += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / count if count > 0 else 0.0


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')
        self.stop      = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  ⚠️  ES counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# PLOT LOSS
# ============================================================
def plot_loss(train_losses, val_losses, save_path, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss',
             color='steelblue', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Val Loss',
             color='coral', marker='o', markersize=3)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.4)

    plt.subplot(1, 2, 2)
    # Smoothed version (rolling average 5 epoch)
    def smooth(values, window=5):
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i+1]) / (i - start + 1))
        return result

    plt.plot(epochs, smooth(train_losses), label='Train (smooth)',
             color='steelblue', linewidth=2)
    plt.plot(epochs, smooth(val_losses), label='Val (smooth)',
             color='coral', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (Smoothed)')
    plt.title(f"{title} — Smoothed")
    plt.legend(); plt.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Grafik: {save_path}")


# ============================================================
# SIMPAN CHECKPOINT
# ============================================================
def save_checkpoint(model, path, info=""):
    torch.save(model.state_dict(), path)
    print(f"  💾 Checkpoint disimpan{': ' + info if info else ''}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    # ── Buat folder output ──
    for fase in ['fase1', 'fase2', 'fase3']:
        os.makedirs(os.path.join(BASE_DIR, "Output_SSD_3Fase", fase), exist_ok=True)
    OUT = os.path.join(BASE_DIR, "Output_SSD_3Fase")

    # ── Load Dataset ──
    print("\n📂 Loading dataset...")
    train_dataset = CocoDataset(TRAIN_IMG, TRAIN_JSON, IMG_SIZE,
                                use_cache=False)  # 8000 gambar → jangan cache
    valid_dataset = CocoDataset(VALID_IMG, VALID_JSON, IMG_SIZE,
                                use_cache=True)   # 500 gambar → aman di-cache

    train_loader = DataLoader(
        train_dataset, batch_size=8,
        shuffle=True,  num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8,
        shuffle=False, num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"✅ Train : {len(train_dataset):,} gambar "
          f"({len(train_loader)} batch/epoch)")
    print(f"✅ Valid : {len(valid_dataset):,} gambar")

    # ── Build Model ──
    model = build_model(NUM_CLASSES, pretrained=True).to(DEVICE)

    # ===========================================================
    # FASE 1 — WARM-UP (30 epoch)
    # Backbone dibekukan, hanya head yang belajar
    # Tujuan: head beradaptasi ke 10 kelas bahan makanan
    #         tanpa merusak fitur pretrained backbone
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"  🥶 FASE 1 — WARM-UP ({EPOCHS_FASE1} epoch)")
    print(f"  Backbone: FROZEN | Head: trainable")
    print(f"{'='*60}")

    for name, param in model.named_parameters():
        param.requires_grad = 'backbone' not in name

    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"  ❄️  Backbone frozen | Head trainable: {n_trainable:,} params")

    optimizer_f1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    # OneCycleLR: warmup → peak → cooldown dalam 1 siklus
    # Sangat efektif untuk konvergensi cepat di awal training
    scheduler_f1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_f1,
        max_lr=1e-3,
        epochs=EPOCHS_FASE1,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,          # 30% epoch pertama untuk naik LR
        div_factor=10,          # LR awal = max_lr / 10
        final_div_factor=1000   # LR akhir = LR awal / 1000
    )

    best_val_f1 = float('inf')
    es1         = EarlyStopping(patience=8, min_delta=1e-4)
    tl1, vl1    = [], []

    for epoch in range(1, EPOCHS_FASE1 + 1):
        tl = train_one_epoch(model, train_loader, optimizer_f1,
                             DEVICE, epoch, ACCUM_STEPS)
        vl = validate(model, valid_loader, DEVICE, epoch)
        scheduler_f1.step()

        tl1.append(tl); vl1.append(vl)
        lr_now = optimizer_f1.param_groups[0]['lr']
        print(f"  Epoch {epoch:02d}/{EPOCHS_FASE1} | "
              f"Train: {tl:.4f} | Val: {vl:.4f} | LR: {lr_now:.2e}")

        if vl < best_val_f1:
            best_val_f1 = vl
            save_checkpoint(
                model,
                os.path.join(OUT, "fase1", "best.pt"),
                f"val_loss={vl:.4f}"
            )

        es1(vl)
        if es1.stop:
            print(f"  🛑 Early stopping epoch {epoch}")
            break

    # Simpan last model fase 1
    save_checkpoint(model, os.path.join(OUT, "fase1", "last.pt"), "last")
    plot_loss(tl1, vl1,
              os.path.join(OUT, "fase1", "loss_curve.png"),
              f"Fase 1 — Warm-up (Backbone Frozen, {len(tl1)} epoch)")
    print(f"\n  ✅ Fase 1 selesai! Best val loss: {best_val_f1:.4f}")

    # ===========================================================
    # FASE 2 — FULL FINE-TUNING (100 epoch)
    # Semua layer dibuka dengan Differential LR
    # Backbone: LR kecil (jaga fitur), Head: LR lebih besar
    # Tujuan: seluruh model menyesuaikan diri ke domain bahan makanan
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"  🔥 FASE 2 — FULL FINE-TUNING ({EPOCHS_FASE2} epoch)")
    print(f"  Backbone: LR=5e-5 | Head: LR=3e-4")
    print(f"{'='*60}")

    # Load model terbaik dari Fase 1
    model.load_state_dict(
        torch.load(os.path.join(OUT, "fase1", "best.pt"),
                   weights_only=False)
    )

    # Buka semua layer kecuali 2 layer paling awal backbone
    # (layer awal hanya deteksi tepi/warna — tidak perlu diubah)
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if any(f'backbone.features.{i}.' in name for i in range(2)):
            param.requires_grad = False

    n_frozen    = sum(p.numel() for p in model.parameters()
                      if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"  ❄️  Frozen (2 layer awal) : {n_frozen:,} params")
    print(f"  🔥 Trainable             : {n_trainable:,} params")

    optimizer_f2 = get_optimizer_diff_lr(
        model, backbone_lr=5e-5, head_lr=3e-4
    )
    # CosineAnnealingWarmRestarts: restart periodik
    # Mencegah model stuck di local minimum
    scheduler_f2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_f2,
        T_0=20,     # restart setiap 20 epoch
        T_mult=2,   # periode restart makin panjang (20 → 40 → 80)
        eta_min=1e-7
    )

    best_val_f2 = float('inf')
    es2         = EarlyStopping(patience=15, min_delta=1e-4)
    tl2, vl2    = [], []

    for epoch in range(1, EPOCHS_FASE2 + 1):
        tl = train_one_epoch(model, train_loader, optimizer_f2,
                             DEVICE, epoch, ACCUM_STEPS)
        vl = validate(model, valid_loader, DEVICE, epoch)
        scheduler_f2.step(epoch)

        tl2.append(tl); vl2.append(vl)
        lr_bb = optimizer_f2.param_groups[0]['lr']
        lr_hd = optimizer_f2.param_groups[1]['lr']
        print(f"  Epoch {epoch:03d}/{EPOCHS_FASE2} | "
              f"Train: {tl:.4f} | Val: {vl:.4f} | "
              f"LR_bb: {lr_bb:.2e} | LR_hd: {lr_hd:.2e}")

        if vl < best_val_f2:
            best_val_f2 = vl
            save_checkpoint(
                model,
                os.path.join(OUT, "fase2", "best.pt"),
                f"val_loss={vl:.4f}"
            )

        # Simpan checkpoint setiap 20 epoch
        if epoch % 20 == 0:
            save_checkpoint(
                model,
                os.path.join(OUT, "fase2", f"checkpoint_e{epoch}.pt"),
                f"epoch={epoch}"
            )

        es2(vl)
        if es2.stop:
            print(f"  🛑 Early stopping epoch {epoch}")
            break

    save_checkpoint(model, os.path.join(OUT, "fase2", "last.pt"), "last")
    plot_loss(tl2, vl2,
              os.path.join(OUT, "fase2", "loss_curve.png"),
              f"Fase 2 — Full Fine-Tuning (Differential LR, {len(tl2)} epoch)")
    print(f"\n  ✅ Fase 2 selesai! Best val loss: {best_val_f2:.4f}")

    # ===========================================================
    # FASE 3 — POLISHING (50 epoch)
    # LR sangat kecil, semua layer terbuka
    # Tujuan: memperhalus bobot, terutama untuk kelas sulit
    #         (Daging Ayam, Tempe, Daging Sapi)
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"  ✨ FASE 3 — POLISHING ({EPOCHS_FASE3} epoch)")
    print(f"  Backbone: LR=1e-5 | Head: LR=5e-5")
    print(f"{'='*60}")

    # Load model terbaik dari Fase 2
    model.load_state_dict(
        torch.load(os.path.join(OUT, "fase2", "best.pt"),
                   weights_only=False)
    )

    # Buka SEMUA layer tanpa terkecuali
    for param in model.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"  🔓 Semua layer terbuka: {n_trainable:,} params")

    optimizer_f3 = get_optimizer_diff_lr(
        model, backbone_lr=1e-5, head_lr=5e-5
    )
    # ReduceLROnPlateau: turunkan LR otomatis jika stagnan
    # Paling cocok untuk fase akhir karena adaptif
    scheduler_f3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_f3,
        mode='min',
        factor=0.5,     # LR × 0.5 jika tidak ada kemajuan
        patience=5,     # tunggu 5 epoch sebelum turunkan LR
        min_lr=1e-8,
        verbose=True
    )

    best_val_f3 = float('inf')
    es3         = EarlyStopping(patience=20, min_delta=1e-5)
    tl3, vl3    = [], []

    for epoch in range(1, EPOCHS_FASE3 + 1):
        tl = train_one_epoch(model, train_loader, optimizer_f3,
                             DEVICE, epoch, ACCUM_STEPS)
        vl = validate(model, valid_loader, DEVICE, epoch)
        scheduler_f3.step(vl)  # ReduceLROnPlateau pakai val_loss

        tl3.append(tl); vl3.append(vl)
        lr_bb = optimizer_f3.param_groups[0]['lr']
        lr_hd = optimizer_f3.param_groups[1]['lr']
        print(f"  Epoch {epoch:02d}/{EPOCHS_FASE3} | "
              f"Train: {tl:.4f} | Val: {vl:.4f} | "
              f"LR_bb: {lr_bb:.2e} | LR_hd: {lr_hd:.2e}")

        if vl < best_val_f3:
            best_val_f3 = vl
            save_checkpoint(
                model,
                os.path.join(OUT, "fase3", "best_final.pt"),
                f"val_loss={vl:.4f}"
            )

        es3(vl)
        if es3.stop:
            print(f"  🛑 Early stopping epoch {epoch}")
            break

    save_checkpoint(model, os.path.join(OUT, "fase3", "last_final.pt"),
                    "last")
    plot_loss(tl3, vl3,
              os.path.join(OUT, "fase3", "loss_curve.png"),
              f"Fase 3 — Polishing (LR sangat kecil, {len(tl3)} epoch)")
    print(f"\n  ✅ Fase 3 selesai! Best val loss: {best_val_f3:.4f}")

    # ===========================================================
    # RINGKASAN AKHIR
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"  🎉 SEMUA FASE TRAINING SELESAI!")
    print(f"{'='*60}")
    # print(f"  Fase 1 best val loss : {best_val_f1:.4f}")
    # print(f"  Fase 2 best val loss : {best_val_f2:.4f}")
    print(f"  Fase 3 best val loss : {best_val_f3:.4f}")
    print(f"\n  📦 Model final untuk evaluasi:")
    print(f"     {os.path.join(OUT, 'fase3', 'best_final.pt')}")
    print(f"\n  📁 Semua output tersimpan di:")
    print(f"     {OUT}")
    print(f"\n  📌 Langkah selanjutnya:")
    print(f"     Jalankan script evaluasi menggunakan best_final.pt")
    print(f"     dan bandingkan dengan hasil SSDLite versi lama.")
    print(f"{'='*60}")