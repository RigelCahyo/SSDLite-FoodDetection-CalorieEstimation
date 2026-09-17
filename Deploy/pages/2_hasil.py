# ============================================================
# pages/2_hasil.py — Hasil Estimasi Kalori
# ============================================================

import streamlit as st
import os, sys, re
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Hasil Estimasi — NutriScan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIWAYAT_PATH = os.path.join(BASE_DIR, "data", "riwayat.csv")

sys.path.append(BASE_DIR)
from calorie           import load_tkpi, hitung_nutrisi, hitung_total
from sidebar_component import render_sidebar, render_bottom_nav, load_css

# Load CSS
load_css()

# ── Fungsi icon ───────────────────────────────────────────────
def make_icon(name, fill="#FBF5DD", stroke="none", sw="2", size="18"):
    path = os.path.join(BASE_DIR, "assets", "icons", f"{name}.svg")
    with open(path) as f:
        svg_str = f.read()
    svg_str  = re.sub(r'<\?xml[^?]*\?>', '', svg_str)
    svg_str  = re.sub(r'<!--.*?-->', '', svg_str, flags=re.DOTALL)
    match_vb = re.search(r'viewBox=["\']([^"\']+)["\']', svg_str)
    vb       = match_vb.group(1) if match_vb else "0 0 24 24"
    match_p  = re.search(r'<svg[^>]*>(.*?)</svg>', svg_str, re.DOTALL)
    paths    = re.sub(r'\s+', ' ', match_p.group(1)).strip() if match_p else ""
    return (
        '<svg viewBox="' + vb + '"'
        ' width="' + size + '" height="' + size + '"'
        ' fill="' + fill + '"'
        ' stroke="' + stroke + '"'
        ' stroke-width="' + sw + '"'
        ' xmlns="http://www.w3.org/2000/svg">'
        + paths +
        '</svg>'
    )

icon_protein = make_icon("protein")
icon_karbo   = make_icon("karbohidrat")
icon_lemak   = make_icon("lemak")
icon_serat   = make_icon("daun")
icon_api     = make_icon("api_estimasi")

# Sidebar
render_sidebar(current_page="hasil")
render_bottom_nav(current_page="hasil")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Hasil Estimasi</h1>
    <p class="page-subtitle">
        Ringkasan kandungan kalori dan nutrisi bahan masakan yang terdeteksi.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Cek session state ─────────────────────────────────────────────────────────
if "input_weights" not in st.session_state or \
   not st.session_state.get("detected_items"):
    st.markdown("""
    <div class="card" style="text-align:center; margin-top:48px; padding:48px;">
        <div style="font-size:16px; font-weight:600;
                    color:#0D530E; margin-bottom:6px;">
            Belum Ada Data Deteksi
        </div>
        <div style="font-size:13px; color:#4a6a4a;">
            Silakan scan atau upload gambar bahan makanan terlebih dahulu.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='btn-kembali'>", unsafe_allow_html=True)
    if st.button("← Ke Halaman Scan", use_container_width=True):
        st.switch_page("pages/1_scan.py")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ── Hitung nutrisi ────────────────────────────────────────────────────────────
berat_inputs   = st.session_state["input_weights"]
detected_items = st.session_state["detected_items"]

df_tkpi  = load_tkpi()
results  = []
for cls_name, berat in berat_inputs.items():
    if berat > 0:
        hasil = hitung_nutrisi(cls_name, berat, df_tkpi)
        if hasil:
            results.append(hasil)

total = hitung_total(results)

# Update total_kalori di riwayat CSV
if os.path.exists(RIWAYAT_PATH):
    try:
        df_r = pd.read_csv(RIWAYAT_PATH)
        if len(df_r) > 0:
            df_r.at[df_r.index[-1], "total_kalori"] = total["energi"]
            df_r.to_csv(RIWAYAT_PATH, index=False)
    except Exception:
        pass

# ── Card 1 — Tabel Estimasi Kalori ───────────────────────────────────────────
total_berat = sum(r["berat"] for r in results)
pct         = min(total["energi"] / 2000 * 100, 100)

tabel_rows = ""
for r in results:
    tabel_rows += (
        '<div class="tabel-row">'
        '<span class="tabel-nama">' + str(r["nama"]) + '</span>'
        '<span class="tabel-berat">' + str(r["berat"]) + ' g</span>'
        '<span class="tabel-kalori">' + str(r["energi"]) + ' kkal</span>'
        '</div>'
    )

html_tabel = (
    '<div class="tabel-wrap">'
        '<div class="tabel-title">'
            '<div class="stat-icon-wrap" style="width:28px;height:28px;border-radius:8px;">'
                + icon_api +
            '</div>'
            'Estimasi Kalori Bahan Masakan'
        '</div>'
        '<div class="tabel-header">'
            '<span>MAKANAN</span>'
            '<span class="th-right">BERAT</span>'
            '<span class="th-right">KALORI</span>'
        '</div>'
        + tabel_rows +
        '<div class="tabel-total-row">'
            '<span class="tabel-total-label">Total</span>'
            '<span class="tabel-total-berat">' + str(total_berat) + ' g</span>'
            '<span class="tabel-total-kalori">' + str(total["energi"]) + ' kkal</span>'
        '</div>'
        '<div class="kalori-progress-wrap">'
            '<div class="kalori-progress-label">'
                '<span>Estimasi Kalori Masakan Jadi (2000 kkal target)</span>'
                '<span>' + str(round(pct)) + '%</span>'
            '</div>'
            '<div class="kalori-progress-bg">'
                '<div class="kalori-progress-fill" style="width:' + str(round(pct, 1)) + '%;"></div>'
            '</div>'
        '</div>'
    '</div>'
)

st.markdown(html_tabel, unsafe_allow_html=True)

# ── Card 2 — Informasi Nutrisi ────────────────────────────────────────────────
AKG = {"protein": 60, "lemak": 67, "karbohidrat": 300, "serat": 28}

def bar_pct(val, ref):
    return min(int(val / ref * 100), 100) if ref > 0 else 0

p_pct = bar_pct(total["protein"],     AKG["protein"])
k_pct = bar_pct(total["karbohidrat"], AKG["karbohidrat"])
l_pct = bar_pct(total["lemak"],       AKG["lemak"])
s_pct = bar_pct(total["serat"],       AKG["serat"])

nutrisi_items = (
    '<div class="nutrisi-item">'
        '<div class="nutrisi-item-header">'
            '<div class="nutrisi-item-left">'
                '<div class="nutrisi-icon">' + icon_protein + '</div>'
                '<span class="nutrisi-label">Protein</span>'
            '</div>'
            '<span class="nutrisi-value">' + str(total['protein']) + '<span>g</span></span>'
        '</div>'
        '<div class="nutrisi-bar-bg">'
            '<div class="nutrisi-bar-fill" style="width:' + str(p_pct) + '%;"></div>'
        '</div>'
    '</div>'
    +
    '<div class="nutrisi-item">'
        '<div class="nutrisi-item-header">'
            '<div class="nutrisi-item-left">'
                '<div class="nutrisi-icon">' + icon_karbo + '</div>'
                '<span class="nutrisi-label">Karbohidrat</span>'
            '</div>'
            '<span class="nutrisi-value">' + str(total['karbohidrat']) + '<span>g</span></span>'
        '</div>'
        '<div class="nutrisi-bar-bg">'
            '<div class="nutrisi-bar-fill" style="width:' + str(k_pct) + '%;"></div>'
        '</div>'
    '</div>'
    +
    '<div class="nutrisi-item">'
        '<div class="nutrisi-item-header">'
            '<div class="nutrisi-item-left">'
                '<div class="nutrisi-icon">' + icon_lemak + '</div>'
                '<span class="nutrisi-label">Lemak</span>'
            '</div>'
            '<span class="nutrisi-value">' + str(total['lemak']) + '<span>g</span></span>'
        '</div>'
        '<div class="nutrisi-bar-bg">'
            '<div class="nutrisi-bar-fill" style="width:' + str(l_pct) + '%;"></div>'
        '</div>'
    '</div>'
    +
    '<div class="nutrisi-item">'
        '<div class="nutrisi-item-header">'
            '<div class="nutrisi-item-left">'
                '<div class="nutrisi-icon">' + icon_serat + '</div>'
                '<span class="nutrisi-label">Serat</span>'
            '</div>'
            '<span class="nutrisi-value">' + str(total['serat']) + '<span>g</span></span>'
        '</div>'
        '<div class="nutrisi-bar-bg">'
            '<div class="nutrisi-bar-fill" style="width:' + str(s_pct) + '%;"></div>'
        '</div>'
    '</div>'
)

html_nutrisi = (
    '<div class="nutrisi-wrap">'
    '<div class="nutrisi-title">Informasi Nutrisi</div>'
    '<div class="nutrisi-grid">'
    + nutrisi_items +
    '</div>'
    '<div class="info-note">'
    '<span>💡</span>'
    '<span>'
    'Nilai nutrisi di atas merupakan estimasi berdasarkan bahan masakan '
    '<strong>mentah</strong> sebelum dimasak. Kandungan nutrisi aktual '
    'dapat berubah tergantung metode memasak yang digunakan.'
    '</span>'
    '</div>'
    '</div>'
)

st.markdown(html_nutrisi, unsafe_allow_html=True)

# ── Tombol Kembali ────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='btn-kembali'>", unsafe_allow_html=True)
if st.button("← Kembali ke Dashboard", use_container_width=True):
    for k in ["detected_items", "input_weights",
              "current_image", "estimation_result"]:
        st.session_state.pop(k, None)
    st.switch_page("app.py")
st.markdown("</div>", unsafe_allow_html=True)