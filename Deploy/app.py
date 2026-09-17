# ============================================================
# app.py — Dashboard Utama
# ============================================================

import streamlit as st
import pandas as pd
import os, sys
from datetime import datetime

st.set_page_config(
    page_title="NutriScan",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BASE_DIR harus didefinisikan PERTAMA
BASE_DIR     = os.path.dirname(__file__)
RIWAYAT_PATH = os.path.join(BASE_DIR, "data", "riwayat.csv")

# Import sidebar_component
sys.path.append(BASE_DIR)
from sidebar_component import render_sidebar, render_bottom_nav, load_css, load_svg, _get_paths, _get_viewbox

# Load CSS
load_css()

# Sidebar
render_sidebar(current_page="dashboard")
render_bottom_nav(current_page="dashboard")

# ── Inisialisasi session state ────────────────────────────────
for key, default in {
    "detected_items"   : {},
    "input_weights"    : {},
    "current_image"    : None,
    "estimation_result": None,
    "input_mode"       : "kamera",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load riwayat ──────────────────────────────────────────────
def load_riwayat():
    if os.path.exists(RIWAYAT_PATH):
        try:
            return pd.read_csv(RIWAYAT_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df_riwayat = load_riwayat()

# Hitung statistik hari ini
scan_hari_ini         = 0
total_kalori_hari_ini = 0.0
total_bahan_hari_ini  = 0

if not df_riwayat.empty and "tanggal" in df_riwayat.columns:
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_today  = df_riwayat[
        df_riwayat["tanggal"].astype(str).str.startswith(today_str)
    ]
    scan_hari_ini = len(df_today)
    if "total_kalori" in df_today.columns:
        total_kalori_hari_ini = df_today["total_kalori"].sum()
    if "jumlah_bahan" in df_today.columns:
        total_bahan_hari_ini  = int(df_today["jumlah_bahan"].sum())

# ── Load SVG ──────────────────────────────────────────────────
bahan_makanan_svg   = load_svg("bahan_makanan")
bahan_makanan_vb    = _get_viewbox(bahan_makanan_svg)
bahan_makanan_paths = _get_paths(bahan_makanan_svg)

bahan_decor_svg     = load_svg("bahan_makanan_decor")
bahan_decor_vb      = _get_viewbox(bahan_decor_svg)
bahan_decor_paths   = _get_paths(bahan_decor_svg)

# ── Konten utama ──────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Selamat Datang! 👋</h1>
    <p class="page-subtitle">
        Deteksi bahan masakan segar dan estimasi kandungan nutrisinya
        secara cerdas dengan teknologi AI.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ────────────────────────────────────────────────
st.markdown(f"""
<div style="display:grid; grid-template-columns:1fr 1fr 1fr;
            gap:12px; margin-bottom:8px;">
    <div class="stat-card">
        <div class="stat-icon-wrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M13 10V3L4 14h7v7l9-11h-7z"
                      stroke="#FBF5DD" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="stat-value">{scan_hari_ini}</div>
        <div class="stat-label">Scan Hari Ini</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon-wrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 2a10 10 0 100 20A10 10 0 0012 2z"
                      stroke="#FBF5DD" stroke-width="2"/>
                <path d="M12 6v6l4 2" stroke="#FBF5DD" stroke-width="2"
                      stroke-linecap="round"/>
            </svg>
        </div>
        <div class="stat-value">{total_kalori_hari_ini:.0f}
            <span style="font-size:14px; font-weight:400;">kkal</span>
        </div>
        <div class="stat-label">Total Kalori</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon-wrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M3 6h18M3 12h18M3 18h18"
                      stroke="#FBF5DD" stroke-width="2"
                      stroke-linecap="round"/>
            </svg>
        </div>
        <div class="stat-value">{total_bahan_hari_ini}</div>
        <div class="stat-label">Bahan Terdeteksi</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ── Feature Banner ────────────────────────────────────────────
st.markdown(f"""
<div class="feature-banner">
    <div class="feature-banner-icon">
        <svg viewBox="{bahan_makanan_vb}" width="44" height="44"
             fill="#FBF5DD" xmlns="http://www.w3.org/2000/svg">
            {bahan_makanan_paths}
        </svg>
    </div>
    <div class="feature-banner-body">
        <div class="feature-banner-title">Teknologi Deteksi Bahan Masakan AI</div>
        <div class="feature-banner-desc">
            Pindai atau unggah foto bahan masakan segar Anda. AI kami akan
            mendeteksi jenis bahan, memperkirakan kandungan nutrisi, dan
            membantu Anda merencanakan masakan secara sehat.
        </div>
        <div class="feature-chips">
            <span class="feature-chip">✓ Akurasi Tinggi</span>
            <span class="feature-chip">✓ Real-time</span>
            <span class="feature-chip">✓ 10 Jenis Bahan</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ── Riwayat Scan ──────────────────────────────────────────────
st.markdown('<div class="section-title">Riwayat Scan Terakhir</div>',
            unsafe_allow_html=True)

if df_riwayat.empty:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:32px; margin-bottom:8px;">📭</div>
        <div style="font-size:14px; color:#306D29;">
            Belum ada riwayat scan. Mulai scan pertamamu!
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    df_show      = df_riwayat.tail(5).iloc[::-1].reset_index(drop=True)
    riwayat_html = ""
    for _, row in df_show.iterrows():
        bahan_list = str(row.get("bahan_terdeteksi", "-"))
        jumlah     = int(row.get("jumlah_bahan", 0))
        tanggal    = str(row.get("tanggal", "-"))
        try:
            dt      = datetime.strptime(tanggal[:19], "%Y-%m-%d %H:%M:%S")
            tgl_fmt = dt.strftime("%H:%M")
        except Exception:
            tgl_fmt = tanggal

        riwayat_html += (
            '<div class="riwayat-item">'
                '<div class="riwayat-icon">'
                    f'<svg viewBox="{bahan_decor_vb}" width="20" height="20" '
                    f'fill="#FBF5DD" xmlns="http://www.w3.org/2000/svg">'
                    f'{bahan_decor_paths}'
                    '</svg>'
                '</div>'
                '<div class="riwayat-body">'
                    '<div class="riwayat-nama">' + bahan_list + '</div>'
                    '<div class="riwayat-waktu">' + tgl_fmt + '</div>'
                '</div>'
                '<div class="riwayat-badge">' + str(jumlah) + ' bahan</div>'
            '</div>'
        )

    st.markdown(
        '<div class="riwayat-wrap">' + riwayat_html + '</div>',
        unsafe_allow_html=True
    )

    col_h1, col_h2 = st.columns([5, 1])
    with col_h2:
        st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
        if st.button("Hapus", use_container_width=True):
            if os.path.exists(RIWAYAT_PATH):
                os.remove(RIWAYAT_PATH)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── CTA Button ────────────────────────────────────────────────
st.markdown("<div class='cta-button'>", unsafe_allow_html=True)
if st.button("Mulai Scan Bahan Makanan", use_container_width=True):
    st.switch_page("pages/1_scan.py")
st.markdown("</div>", unsafe_allow_html=True)