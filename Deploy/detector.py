# ============================================================
# detector.py — Logika Deteksi SSDLite MobileNetV3
# ============================================================

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ============================================================
# KONFIGURASI
# ============================================================
NUM_CLASSES = 11
IMG_SIZE    = 320
CONF_THRESH = 0.3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "model/best_final.pt"

CLASS_NAMES = [
    "background",
    "Bayam", "Brokoli", "Daging Ayam", "Daging Sapi",
    "Kentang", "Tahu", "Telur", "Tempe", "Tomat", "Wortel"
]

CLASS_COLORS = {
    "Bayam"      : (46,  204, 113),
    "Brokoli"    : (39,  174, 96 ),
    "Daging Ayam": (230, 126, 34 ),
    "Daging Sapi": (192, 57,  43 ),
    "Kentang"    : (241, 196, 15 ),
    "Tahu"       : (149, 165, 166),
    "Telur"      : (243, 156, 18 ),
    "Tempe"      : (139, 90,  43 ),
    "Tomat"      : (231, 76,  60 ),
    "Wortel"     : (230, 100, 34 ),
}

# ============================================================
# LOAD MODEL
# ============================================================
@torch.no_grad()
@torch.no_grad()
def load_model():
    # Load dengan weights DEFAULT dulu — arsitektur harus identik dengan saat training
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')

    # Ganti classification head sesuai jumlah kelas kita (sama persis seperti training)
    in_channels = det_utils.retrieve_out_channels(
        model.backbone, (IMG_SIZE, IMG_SIZE)
    )
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=torch.nn.BatchNorm2d
    )

    # Baru load bobot hasil training
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    )

    model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)

# ============================================================
# DETEKSI
# ============================================================
def detect(model, img_pil, conf_thresh=CONF_THRESH):
    img_tensor = preprocess(img_pil)

    with torch.no_grad():
        preds = model(img_tensor)

    pred   = preds[0]
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    boxes  = pred['boxes'].cpu().numpy()

    keep   = scores >= conf_thresh
    return boxes[keep], labels[keep], scores[keep]

# ============================================================
# AMBIL KELAS UNIK
# (jika 2 tempe terdeteksi → tampil 1 "Tempe" saja)
# ============================================================
def get_unique_classes(labels, scores):
    seen = {}
    for lbl, score in zip(labels, scores):
        if lbl >= len(CLASS_NAMES):
            continue
        cls_name = CLASS_NAMES[lbl]
        if cls_name == "background":
            continue
        # Simpan hanya confidence tertinggi per kelas
        if cls_name not in seen or score > seen[cls_name]:
            seen[cls_name] = float(score)
    return seen  # {nama_kelas: confidence_tertinggi}

# ============================================================
# GAMBAR BOUNDING BOX
# ============================================================
def draw_boxes(img_pil, boxes, labels, scores):
    img_draw       = img_pil.copy()
    draw           = ImageDraw.Draw(img_draw)
    orig_w, orig_h = img_pil.size

    try:
        font       = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font       = ImageFont.load_default()
        font_small = font

    for box, lbl, score in zip(boxes, labels, scores):
        if lbl >= len(CLASS_NAMES):
            continue
        cls_name = CLASS_NAMES[lbl]
        if cls_name == "background":
            continue

        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Skala box dari IMG_SIZE ke ukuran asli
        x1 = int(box[0] / IMG_SIZE * orig_w)
        y1 = int(box[1] / IMG_SIZE * orig_h)
        x2 = int(box[2] / IMG_SIZE * orig_w)
        y2 = int(box[3] / IMG_SIZE * orig_h)

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        label_text = f"{cls_name} {score:.2f}"
        bbox_text  = draw.textbbox((x1, y1), label_text, font=font)
        text_w     = bbox_text[2] - bbox_text[0]
        text_h     = bbox_text[3] - bbox_text[1]
        draw.rectangle(
            [x1, y1 - text_h - 6, x1 + text_w + 6, y1],
            fill=color
        )
        draw.text((x1 + 3, y1 - text_h - 3), label_text,
                  fill="white", font=font)

    return img_draw