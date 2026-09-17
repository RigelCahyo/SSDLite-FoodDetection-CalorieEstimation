# ============================================================
# sidebar_component.py
# ============================================================

import streamlit as st
import os
import re

BASE_DIR = os.path.dirname(__file__)

def load_css():
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_svg(name):
    path = os.path.join(BASE_DIR, "assets", "icons", f"{name}.svg")
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        content = re.sub(r'<\?xml[^?]*\?>', '', content)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        return content.strip()
    return ""

def _get_viewbox(svg_str):
    match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_str)
    return match.group(1) if match else "0 0 24 24"

def _get_paths(svg_str):
    match = re.search(r'<svg[^>]*>(.*?)</svg>', svg_str, re.DOTALL)
    return match.group(1).strip() if match else ""

def render_sidebar(current_page="dashboard", model_ready=None):

    # ── Cek query param untuk navigasi ──
    params = st.query_params
    if "nav" in params:
        nav_target = params["nav"]
        st.query_params.clear()
        if nav_target == "home":
            st.switch_page("app.py")
        elif nav_target == "scan":
            st.switch_page("pages/1_scan.py")
        elif nav_target == "hasil":
            st.switch_page("pages/2_hasil.py")
        elif nav_target == "info":
            st.switch_page("pages/3_informasi.py")

    home_active  = "nav-active" if current_page == "dashboard" else ""
    scan_active  = "nav-active" if current_page == "scan"      else ""
    hasil_active = "nav-active" if current_page == "hasil"     else ""
    info_active  = "nav-active" if current_page == "info"      else ""

    # Load icon daun untuk logo
    daun_svg = load_svg("daun")

    # ── Build logo_icon ──
    if daun_svg:
        vb    = _get_viewbox(daun_svg)
        paths = _get_paths(daun_svg)
        logo_icon = (
            '<div style="width:22px; height:22px; overflow:hidden;'
            ' display:flex; align-items:center; justify-content:center;">'
            '<svg viewBox="' + vb + '" width="22" height="22"'
            ' fill="#FBF5DD" xmlns="http://www.w3.org/2000/svg">'
            + paths +
            '</svg>'
            '</div>'
        )
    else:
        logo_icon = (
            '<svg width="22" height="22" viewBox="0 0 24 24" fill="#FBF5DD">'
            '<path d="M7 10c0-2.76 2.24-5 5-5s5 2.24 5 5'
            'c0 2.33-1.6 4.3-3.75 4.87V17h-2.5v-2.13'
            'C8.6 14.3 7 12.33 7 10z"/>'
            '</svg>'
        )

    # ── Build status_html ──
    if model_ready is not None:
        status_color = "#4ade80" if model_ready else "#f87171"
        status_text  = "Model Siap" if model_ready else "Model Tidak Ditemukan"
        status_html  = (
            '<div style="margin: 20px 16px 0; padding: 10px 14px;'
            ' background: rgba(255,255,255,0.06); border-radius: 10px;'
            ' display: flex; align-items: center; gap: 8px;">'
            '<div style="width:8px; height:8px; border-radius:50%;'
            ' background:' + status_color + '; flex-shrink:0;"></div>'
            '<span style="font-size:12px; color:rgba(251,245,221,0.7);'
            ' font-family:\'Poppins\',sans-serif;">'
            + status_text +
            '</span>'
            '</div>'
        )
    else:
        status_html = ""

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">'
            '<div class="sidebar-logo-wrap">'
            + logo_icon +
            '</div>'
            '<div class="sidebar-title-wrap">'
            '<div class="sidebar-app-name">NutriScan</div>'
            '<div class="sidebar-app-sub">Ingredient Detection AI</div>'
            '</div>'
            '</div>'
            '<div class="sidebar-divider"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<a href="?nav=home" target="_self" class="nav-item ' + home_active + '">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"'
            ' stroke="currentColor" stroke-width="2"'
            ' stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"/>'
            '<path d="M9 21V12h6v9"/>'
            '</svg>'
            '<span>Dashboard</span>'
            '</a>'
            '<a href="?nav=scan" target="_self" class="nav-item ' + scan_active + '">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"'
            ' stroke="currentColor" stroke-width="2"'
            ' stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="3" y="3" width="7" height="7" rx="1"/>'
            '<rect x="14" y="3" width="7" height="7" rx="1"/>'
            '<rect x="3" y="14" width="7" height="7" rx="1"/>'
            '<circle cx="17" cy="17" r="3"/>'
            '</svg>'
            '<span>Scan &amp; Deteksi</span>'
            '</a>'
            '<a href="?nav=hasil" target="_self" class="nav-item ' + hasil_active + '">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"'
            ' stroke="currentColor" stroke-width="2"'
            ' stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="18" y1="20" x2="18" y2="10"/>'
            '<line x1="12" y1="20" x2="12" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="14"/>'
            '</svg>'
            '<span>Hasil Estimasi</span>'
            '</a>'
            '<a href="?nav=info" target="_self" class="nav-item ' + info_active + '">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"'
            ' stroke="currentColor" stroke-width="2"'
            ' stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="8" x2="12" y2="8.5"'
            ' stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="12" y1="12" x2="12" y2="16"/>'
            '</svg>'
            '<span>Informasi</span>'
            '</a>',
            unsafe_allow_html=True
        )

        if status_html:
            st.markdown(status_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="sidebar-footer">v1.0.0 · Deteksi Bahan Masakan</div>',
            unsafe_allow_html=True
        )


def render_bottom_nav(current_page="dashboard"):
    home_active  = "active" if current_page == "dashboard" else ""
    scan_active  = "active" if current_page == "scan"      else ""
    hasil_active = "active" if current_page == "hasil"     else ""
    info_active  = "active" if current_page == "info"      else ""

    st.markdown(
        '<div class="bottom-nav">'
        '<a href="?nav=home" target="_self" class="bottom-nav-item ' + home_active + '">'
        '<span style="font-size:20px;">🏠</span>'
        '<span>Home</span>'
        '</a>'
        '<a href="?nav=scan" target="_self" class="bottom-nav-item ' + scan_active + '">'
        '<span style="font-size:20px;">📷</span>'
        '<span>Scan</span>'
        '</a>'
        '<a href="?nav=hasil" target="_self" class="bottom-nav-item ' + hasil_active + '">'
        '<span style="font-size:20px;">📊</span>'
        '<span>Hasil</span>'
        '</a>'
        '<a href="?nav=info" target="_self" class="bottom-nav-item ' + info_active + '">'
        '<span style="font-size:20px;">ℹ️</span>'
        '<span>Info</span>'
        '</a>'
        '</div>',
        unsafe_allow_html=True
    )