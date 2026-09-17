# ============================================================
# pages/1_scan.py — Scan & Deteksi Bahan
# ============================================================

import streamlit as st
from PIL import Image
import os, sys
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="Scan & Deteksi — NutriScan",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIWAYAT_PATH = os.path.join(BASE_DIR, "data", "riwayat.csv")
sys.path.append(BASE_DIR)

from detector          import load_model, detect, get_unique_classes, draw_boxes
from sidebar_component import render_sidebar, render_bottom_nav, load_css, load_svg, _get_paths, _get_viewbox
from toggle_component  import render_toggle

load_css()

# Cek model
model_path  = os.path.join(BASE_DIR, "model", "best_final.pt")
model_ready = os.path.exists(model_path)

# Sidebar
render_sidebar(current_page="scan", model_ready=model_ready)
render_bottom_nav(current_page="scan")

# ── Session state ─────────────────────────────────────────────
for key, val in {
    "detected_items": {},
    "input_weights": {},
    "current_image": None,
    "estimation_result": None,
    "input_mode": "kamera",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model deteksi...")
def get_model():
    return load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Scan & Deteksi Bahan</h1>
    <p class="page-subtitle">
        Pindai atau unggah foto bahan masakan segar untuk identifikasi otomatis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Toggle ────────────────────────────────────────────────────
render_toggle(
    current_mode=st.session_state.input_mode,
    base_dir=BASE_DIR
)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── Area input gambar ─────────────────────────────────────────
img_pil = None

if st.session_state.input_mode == "kamera":
    st.markdown("""
    <div style="text-align:center; margin-bottom:10px;">
        <div style="display:inline-flex; align-items:center; gap:8px;
                    background:#E8E4C4; border-radius:20px; padding:6px 16px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4
                         l2-3h6l2 3h4a2 2 0 012 2z"
                      stroke="#306D29" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="13" r="4"
                        stroke="#306D29" stroke-width="2"/>
            </svg>
            <span style="font-family:Poppins,sans-serif; font-size:13px;
                         font-weight:600; color:#1a5c1a;">
                Arahkan kamera ke bahan masakan Anda
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    camera_img = st.camera_input(
        "Ambil foto",
        label_visibility="collapsed"
    )
    if camera_img:
        img_pil = Image.open(camera_img).convert("RGB")

else:
    uploaded = st.file_uploader(
        "Seret gambar ke sini atau klik untuk memilih",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")

# ── Proses deteksi ────────────────────────────────────────────
if img_pil is not None:

    if not model_ready:
        st.error(
            "❌ File `best_final.pt` tidak ditemukan di folder `model/`. "
            "Pastikan model sudah tersedia."
        )
        st.stop()

    model = get_model()

    conf_thresh = 0.3

    col_img1, col_img2 = st.columns(2, gap="medium")

    with col_img1:
        st.markdown("""
        <div style='font-size:13px; font-weight:600;
                    color:#4a6a4a; margin-bottom:6px;'>
            Gambar Input
        </div>
        """, unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

    with st.spinner("🔍 Mendeteksi bahan makanan..."):
        boxes, labels, scores = detect(model, img_pil, conf_thresh)
        unique_classes        = get_unique_classes(labels, scores)
        img_result            = draw_boxes(img_pil, boxes, labels, scores)

    with col_img2:
        st.markdown("""
        <div style='font-size:13px; font-weight:600;
                    color:#4a6a4a; margin-bottom:6px;'>
            Hasil Deteksi
        </div>
        """, unsafe_allow_html=True)
        st.image(img_result, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Hasil deteksi ─────────────────────────────────────────
    if not unique_classes:
        st.markdown("""
        <div class="card" style="text-align:center; padding:32px;">
            <div style="font-size:28px; margin-bottom:8px;">🔍</div>
            <div style="font-size:14px; color:#4a6a4a;">
                Belum ada bahan makanan yang terdeteksi.<br>
                <i>Pastikan gambar cukup jelas dan terdapat bahan makanan yang terlihat untuk dianalisis.<i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="section-title" style="margin-top:4px;">
            Bahan Terdeteksi — {len(unique_classes)} item
        </div>
        """, unsafe_allow_html=True)

        for cls_name, conf in unique_classes.items():
            bar_w = int(conf * 100)
            st.markdown(f"""
            <div class="det-item">
                <div class="det-header">
                    <span class="det-name">{cls_name}</span>
                    <span class="det-conf">{conf:.1%}</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_w}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">Masukkan Berat Bahan (gram)</div>
        """, unsafe_allow_html=True)

        # ── Input berat — pakai number_input bawaan Streamlit ─
        berat_inputs = {}
        n    = len(unique_classes)
        cols = st.columns(min(n, 3), gap="medium")

        for i, (cls_name, conf) in enumerate(unique_classes.items()):
            with cols[i % min(n, 3)]:
                st.markdown(f"""
                <div style='font-size:13px; font-weight:600;
                            color:#0D530E; margin-bottom:4px;'>
                    {cls_name}
                </div>
                """, unsafe_allow_html=True)
                berat_inputs[cls_name] = st.number_input(
                    label=cls_name,
                    min_value=0, max_value=5000,
                    value=100, step=10,
                    key=f"berat_{cls_name}",
                    label_visibility="collapsed"
                )

        st.session_state.detected_items = unique_classes
        st.session_state.input_weights  = berat_inputs
        st.session_state.current_image  = img_result

# ── Tombol Lanjut ke Hasil ────────────────────────────────────
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='btn-lanjut'>", unsafe_allow_html=True)
if st.button("Lanjut ke Hasil Estimasi →", use_container_width=True):
    if not st.session_state.detected_items:
        st.error("⚠️ Belum ada bahan terdeteksi. Scan atau upload gambar terlebih dahulu.")
    elif all(v == 0 for v in st.session_state.input_weights.values()):
        st.error("⚠️ Masukkan berat minimal 1 bahan makanan!")
    else:
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        nama_bahan = ", ".join(st.session_state.detected_items.keys())
        new_row = {
            "tanggal":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bahan_terdeteksi": nama_bahan,
            "jumlah_bahan":     len(st.session_state.detected_items),
            "total_kalori":     0,
        }
        if os.path.exists(RIWAYAT_PATH):
            df = pd.read_csv(RIWAYAT_PATH)
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(RIWAYAT_PATH, index=False)
        st.switch_page("pages/2_hasil.py")
st.markdown("</div>", unsafe_allow_html=True)