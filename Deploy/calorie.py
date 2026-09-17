# ============================================================
# calorie.py — Kalkulasi Kalori dari TKPI
# ============================================================

import pandas as pd
import os

TKPI_PATH = "data/tkpi.xlsx"

# ============================================================
# LOAD DATA TKPI
# ============================================================
def load_tkpi():
    df = pd.read_excel(TKPI_PATH)
    # Bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()
    return df

# ============================================================
# HITUNG KALORI & NUTRISI
# ============================================================
def hitung_nutrisi(nama_bahan, berat_gram, df_tkpi):
    """
    Hitung estimasi nutrisi berdasarkan berat (gram).
    Semua nilai di TKPI per 100 gram BDD.
    Karena BDD semua 100%, rumus: (berat / 100) × nilai_per_100g
    """
    row = df_tkpi[
        df_tkpi['Nama Bahan Makanan'].str.strip().str.lower()
        == nama_bahan.strip().lower()
    ]

    if row.empty:
        return None

    row    = row.iloc[0]
    faktor = berat_gram / 100

    def safe_val(col):
        val = row.get(col, 0)
        try:
            val = float(val)
            return round(faktor * val, 1)
        except:
            return 0.0

    return {
        "nama"        : nama_bahan,
        "berat"       : berat_gram,
        "energi"      : safe_val("Energi (Kal)"),
        "protein"     : safe_val("Protein (g)"),
        "lemak"       : safe_val("Lemak (g)"),
        "karbohidrat" : safe_val("Karbohidrat (g)"),
        "serat"       : safe_val("Serat (g)"),
    }

# ============================================================
# HITUNG TOTAL SEMUA BAHAN
# ============================================================
def hitung_total(results):
    total = {
        "energi"     : 0.0,
        "protein"    : 0.0,
        "lemak"      : 0.0,
        "karbohidrat": 0.0,
        "serat"      : 0.0,
    }
    for r in results:
        if r:
            for key in total:
                total[key] += r[key]

    return {k: round(v, 1) for k, v in total.items()}