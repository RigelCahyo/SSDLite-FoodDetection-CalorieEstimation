# ============================================================
# toggle_component.py
# ============================================================
import streamlit as st
import os

def render_toggle(current_mode="kamera", base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Cek query param untuk switch mode
    params = st.query_params
    if "mode" in params:
        mode_target = params["mode"]
        st.query_params.clear()
        st.session_state.input_mode = mode_target
        st.rerun()

    # Load SVG
    def load_svg(name):
        path = os.path.join(base_dir, "assets", "icons", f"{name}.svg")
        with open(path) as f:
            return f.read()

    def color_svg(svg, color):
        return svg.replace('stroke="#FBF5DD"', f'stroke="{color}"')\
                  .replace("stroke='#FBF5DD'", f"stroke='{color}'")\
                  .replace('fill="#FBF5DD"',   f'fill="{color}"')\
                  .replace("fill='#FBF5DD'",   f"fill='{color}'")

    kamera_svg = load_svg("kamera")
    upload_svg = load_svg("upload")

    k_active = current_mode == "kamera"
    u_active = current_mode == "upload"

    k_bg     = "#1a5c1a" if k_active else "#F5F0DC"
    k_border = "2px solid transparent"    if k_active else "2px solid #306D29"
    k_color  = "#FBF5DD" if k_active else "#306D29"

    u_bg     = "#1a5c1a" if u_active else "#F5F0DC"
    u_border = "2px solid transparent"    if u_active else "2px solid #306D29"
    u_color  = "#FBF5DD" if u_active else "#306D29"

    k_svg = color_svg(kamera_svg, k_color)
    u_svg = color_svg(upload_svg, u_color)

    col_t1, col_t2 = st.columns(2, gap="small")

    with col_t1:
        st.markdown(f"""
        <a href="?mode=kamera" target="_self"
           style="background:{k_bg}; border:{k_border};
                  border-radius:10px; padding:10px 16px;
                  display:flex; align-items:center; justify-content:center;
                  gap:8px; text-decoration:none; cursor:pointer;">
            {k_svg}
            <span style="font-family:Poppins,sans-serif; font-size:0.9rem;
                         font-weight:600; color:{k_color};">
                Scan Langsung
            </span>
        </a>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <a href="?mode=upload" target="_self"
           style="background:{u_bg}; border:{u_border};
                  border-radius:10px; padding:10px 16px;
                  display:flex; align-items:center; justify-content:center;
                  gap:8px; text-decoration:none; cursor:pointer;">
            {u_svg}
            <span style="font-family:Poppins,sans-serif; font-size:0.9rem;
                         font-weight:600; color:{u_color};">
                Upload Gambar
            </span>
        </a>
        """, unsafe_allow_html=True)