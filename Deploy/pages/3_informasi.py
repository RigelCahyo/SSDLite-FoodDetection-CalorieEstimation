# ============================================================
# pages/3_informasi.py — Informasi Pembuat & Aplikasi
# ============================================================

import streamlit as st
import os, sys

st.set_page_config(
    page_title="Informasi — NutriScan",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BASE_DIR dan import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)
from sidebar_component import render_sidebar, render_bottom_nav, load_css

# Load CSS
load_css()

# Sidebar — tanpa model_ready
render_sidebar(current_page="info")
render_bottom_nav(current_page="info")

# ... sisa kode tidak berubah
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Informasi Pembuat</h1>
    <p class="page-subtitle">Detail tentang pembuat dan aplikasi NutriScan.</p>
</div>
""", unsafe_allow_html=True)

# ── Tengahkan dengan columns ──────────────────────────────────
_, col_mid, _ = st.columns([1, 2, 1])

with col_mid:

    # ── Foto profil dengan border hijau ──────────────────────
    foto_path = os.path.join(BASE_DIR, "assets", "foto", "image.png")

    if os.path.exists(foto_path):
        # Konversi foto ke base64 agar bisa di-embed dalam HTML
        import base64
        with open(foto_path, "rb") as f:
            foto_b64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <div style="display:flex; justify-content:center;
                    margin-bottom:20px;">
            <img src="data:image/jpeg;base64,{foto_b64}"
                 style="width:160px; height:160px;
                        border-radius:50%;
                        object-fit:cover;
                        border: 4px solid #306D29;
                        box-shadow: 0 0 0 6px rgba(48,109,41,0.15);" />
        </div>
        """, unsafe_allow_html=True)
    else:
        # Placeholder jika foto belum ada
        st.markdown("""
        <div style="display:flex; justify-content:center;
                    margin-bottom:20px;">
            <div style="width:160px; height:160px;
                        border-radius:50%;
                        border: 4px solid #306D29;
                        box-shadow: 0 0 0 6px rgba(48,109,41,0.15);
                        background:#E8E4C4;
                        display:flex; align-items:center;
                        justify-content:center;
                        font-size:52px;">
                👨‍🎓
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Card profil ───────────────────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding:32px 28px;">
        <div style="font-size:20px; font-weight:700;
                    color:#0D530E; margin-bottom:4px;">
            Rigel Cahyo Gumilang Susanto
        </div>
        <div style="font-size:13px; color:#4a6a4a; margin-bottom:16px;">
            NPM: 5221811013
        </div>
        <div style="height:1px; background:#ddd8b8; margin:0 0 16px;"></div>
        <div style="font-size:13px; color:#3a5a3a;
                    line-height:1.9; text-align:left;">
            <b>Program Studi:</b> Sains Data<br>
            <b>Universitas:</b> Universitas Teknologi Yogyakarta<br>
            <b>Tahun:</b> 2025 / 2026
        </div>
        <div style="height:1px; background:#ddd8b8; margin:16px 0;"></div>
        <div style="font-size:12px; color:#4a6a4a;
                    font-style:italic; margin-bottom:6px;">
            Judul Skripsi:
        </div>
        <div style="font-size:14px; font-weight:600; color:#0D530E;">
            "Implementasi Single Shot Multibox Detector Lite (SSDLite) MobileNetv3 Untuk Deteksi Bahan Makanan Dan Estimasi Kandungan Kalori Berbasis Data Citra"
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Card tentang aplikasi ─────────────────────────────────
    rows = [
        ("Model",      "SSDLite MobileNetV3-Large"),
        ("Framework",  "PyTorch + Streamlit"),
        ("Jumlah Kelas",      "10 Bahan Makanan"),
        ("Dataset Citra",    "9.000 Gambar (COCO Format)"),
        ("Data Nutrisi",    "TKPI 2019"),
    ]

    rows_html = "".join(f"""
    <div style="display:flex; align-items:center; padding:10px 0;
                border-bottom:1px solid #ddd8b8;">
        <span style="width:160px; font-size:13px;
                     color:#4a6a4a; flex-shrink:0;">{label}</span>
        <span style="font-size:13px; font-weight:500;
                     color:#0D530E;">{value}</span>
    </div>
    """ for label, value in rows)

    st.markdown(f"""
    <div class="card" style="padding:24px 28px;">
        <div style="font-size:16px; font-weight:700;
                    color:#0D530E; margin-bottom:12px;">
            🤖 Tentang Aplikasi
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Tombol kembali ────────────────────────────────────────
    st.markdown("<div class='btn-kembali'>", unsafe_allow_html=True)
    if st.button("← Kembali ke Dashboard", use_container_width=True):
        st.switch_page("app.py")
    st.markdown("</div>", unsafe_allow_html=True)