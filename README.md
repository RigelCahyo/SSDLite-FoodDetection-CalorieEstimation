SSDLite-FoodDetection-CalorieEstimation
Food ingredient detection (SSDLite MobileNetV3) with automatic calorie &amp; nutrition estimation via Streamlit — undergraduate thesis project.
# NutriScan — Deteksi Bahan Makanan dan Estimasi Kalori Berbasis SSDLite MobileNetV3

> Sistem deteksi objek untuk mengidentifikasi 10 jenis bahan makanan mentah dari citra menggunakan arsitektur **SSDLite MobileNetV3-Large**, diintegrasikan dengan **Tabel Komposisi Pangan Indonesia (TKPI)** untuk menghasilkan estimasi kalori dan kandungan nutrisi secara otomatis melalui aplikasi web **Streamlit**.

## Deskripsi Proyek

Permasalahan gizi di Indonesia bersifat ganda: prevalensi *stunting* nasional tercatat 19,8% pada 2024 (turun dari 21,5% di 2023), namun di sisi lain ketidakseimbangan asupan kalori juga mendorong peningkatan obesitas dan penyakit tidak menular pada kelompok usia produktif. Peluncuran Program Makan Bergizi Gratis (MBG) pada Januari 2025 turut mendorong kesadaran masyarakat terhadap pentingnya pemantauan asupan kalori harian — namun proses estimasi kalori secara manual melalui tabel komposisi pangan masih memakan waktu, rawan salah input, dan mengharuskan pengguna memiliki pengetahuan gizi yang memadai.

Proyek ini merupakan **Laporan Tugas Akhir**, Program Studi Sains Data, Fakultas Sains & Teknologi, Universitas Teknologi Yogyakarta (2026), yang menjawab masalah tersebut dengan membangun sistem *computer vision* ringan (mampu berjalan pada perangkat dengan sumber daya terbatas) untuk mendeteksi bahan makanan mentah dari citra, lalu mengonversi hasil deteksi tersebut menjadi estimasi kalori dan nutrisi berdasarkan berat yang diinput pengguna.

**Rumusan Masalah:**
1. Bagaimana mengetahui kandungan kalori bahan makanan berdasarkan citra dengan memanfaatkan model SSDLite MobileNetV3 untuk mengidentifikasi jenis bahan makanan yang terdeteksi?
2. Bagaimana kinerja model SSDLite MobileNetV3 dalam mengidentifikasi bahan makanan jika diukur menggunakan *Precision*, *Recall*, F1-Score, dan mAP (IoU threshold 0,5)?
3. Bagaimana mekanisme integrasi hasil deteksi dengan basis data TKPI untuk menampilkan kandungan kalori dan nutrisi secara proporsional berdasarkan berat input manual pengguna?

**Cakupan:** 10 kelas bahan makanan mentah — bayam, brokoli, daging ayam, daging sapi, kentang, tahu, telur, tempe, tomat, dan wortel. Sistem **tidak** mencakup makanan olahan, makanan siap saji, bahan yang membusuk, maupun hidangan campuran dalam satu piring.

## Dataset

| Aspek | Keterangan |
|---|---|
| Sumber citra | Dataset publik *Roboflow* + pengambilan foto mandiri (kamera smartphone) |
| Sumber data kalori | Tabel Komposisi Pangan Indonesia (TKPI), Kementerian Kesehatan RI |
| Jumlah citra | 5.000 citra, 500 citra/kelas × 10 kelas |
| Format anotasi | COCO JSON (bounding box), dianotasi manual via Roboflow |
| Rasio split | 80:10:10 (latih:validasi:uji) dengan *stratified splitting* |
| Preprocessing | Auto-orient, resize *fit (black edges)* ke 320×320 px, normalisasi ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| Augmentasi | Dilakukan *offline* via Roboflow, diterapkan hanya pada data latih |

Data kalori (`tkpi.xlsx`) memuat kolom **Energi (Kal), Protein (g), Lemak (g), Karbohidrat (g), Serat (g)** per 100 gram untuk masing-masing dari 10 bahan makanan, dengan asumsi BDD (Bagian Dapat Dimakan) 100%.

## Metodologi

Alur penelitian terdiri atas 3 fase besar dengan 11 tahapan utama:

1. **Studi Literatur** — kajian *Computer Vision*, deteksi objek, SSDLite MobileNetV3.
2. **Identifikasi Permasalahan** — celah penelitian pada deteksi bahan mentah (bukan makanan jadi).
3. **Pengumpulan Data** — Roboflow + foto mandiri.
4. **Cleaning Data** — eliminasi gambar buram, tidak relevan, atau duplikat.
5. **Labelling Data** — anotasi *bounding box* manual via Roboflow.
6. **Splitting Data** — 80:10:10, *stratified*.
7. **Preprocessing** — auto-orient, resize 320×320, normalisasi tensor.
8. **Augmentasi Data** — offline via Roboflow, hanya pada data latih.
9. **Training Model** — strategi 3 fase (*warm-up → fine-tuning → polishing*), *transfer learning* dari bobot COCO, *gradient accumulation* untuk mengatasi keterbatasan VRAM.
10. **Evaluasi Model** — mAP@0.5, Precision, Recall, F1-Score pada data uji per fase.
11. **Deployment** — integrasi bobot final ke aplikasi Streamlit "NutriScan".

##  Model dan Evaluasi

**Arsitektur:** SSDLite320 MobileNetV3-Large (`torchvision`), *classification head* diganti dari 91 kelas COCO → 11 kelas (10 bahan makanan + background), menggunakan bobot *pretrained* COCO sebagai titik awal.

**Strategi training — 3 fase, total maksimum 180 epoch:**

| Konfigurasi | Fase 1 (Warm-up) | Fase 2 (Fine-Tuning) | Fase 3 (Polishing) |
|---|---|---|---|
| Max epoch | 30 | 100 | 50 |
| Layer dilatih | Hanya *head* (backbone frozen) | Hampir semua layer (2 layer awal backbone tetap frozen) | Seluruh layer |
| Optimizer | AdamW (lr 1×10⁻³) | AdamW, differential LR (backbone 5×10⁻⁵, head 3×10⁻⁴) | AdamW, differential LR (backbone 1×10⁻⁵, head 5×10⁻⁵) |
| Scheduler | OneCycleLR | CosineAnnealingWarmRestarts (T₀=20, T_mult=2) | ReduceLROnPlateau (factor 0.5, patience 5) |
| Early stopping | patience 8 | patience 15 | patience 20 |

*Gradient accumulation* (4 steps) diterapkan di seluruh fase sehingga *batch* efektif = 32 (dari *batch* aktual 8), dikombinasikan dengan *gradient clipping* (max norm 3.0), untuk mengatasi keterbatasan VRAM GPU (NVIDIA RTX 3050 4GB).

**Hasil evaluasi** (data uji 500 citra, *confidence threshold* 0,3, IoU *threshold* 0,5):

| Fase | Strategi | mAP@0.5 |
|---|---|---|
| Fase 1 | Warm-up (Backbone Frozen) | 0,7098 |
| Fase 2 | Full Fine-Tuning (Differential LR) | 0,7655 |
| **Fase 3 (Final)** | **Polishing (LR sangat kecil)** | **0,8445** |

**Metrik per kelas — Model Final (Fase 3):**

| Kelas | AP | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Tomat | 1,000 | 0,978 | 1,000 | 0,989 |
| Kentang | 0,997 | 0,941 | 1,000 | 0,969 |
| Wortel | 0,897 | 0,731 | 0,980 | 0,838 |
| Brokoli | 0,895 | 0,690 | 0,952 | 0,800 |
| Daging Sapi | 0,862 | 0,612 | 0,984 | 0,755 |
| Tahu | 0,865 | 0,760 | 0,905 | 0,826 |
| Telur | 0,877 | 0,735 | 0,943 | 0,826 |
| Tempe | 0,767 | 0,533 | 0,859 | 0,658 |
| Bayam | 0,655 | 0,539 | 0,724 | 0,618 |
| Daging Ayam | 0,631 | 0,455 | 0,823 | 0,586 |

Nilai mAP@0.5 model final (0,8445) telah melampaui target performansi minimum yang ditetapkan (0,80), sehingga model dikategorikan memenuhi kebutuhan performansi sistem.

##  Insight / Analisis

- **Kelas terbaik:** Tomat dan Wortel — warna solid/khas dan bentuk konsisten memudahkan model membedakannya dari latar belakang.
- **Kelas tersulit:** Daging Ayam dan Bayam — kemiripan visual tinggi dengan kelas lain (misal jeroan ayam vs daging sapi karena warna merah keunguan serupa; bayam vs kale karena tekstur daun hijau serupa).
- **Pola Precision vs Recall:** hampir seluruh kelas memiliki Recall > Precision, karakteristik khas arsitektur SSD berbasis *anchor* yang cenderung *over-detecting* (banyak kandidat box), sehingga risiko *false positive* lebih tinggi dibanding *false negative*.
- **Kontribusi tiap fase:** peningkatan terbesar terjadi pada transisi Fase 2 → Fase 3 (+0,079), lebih besar dari Fase 1 → Fase 2 (+0,056) — menunjukkan tahap *polishing* dengan LR sangat kecil memberi penyempurnaan paling signifikan, terutama pada kelas bertekstur kompleks (Tempe +0,201, Daging Sapi +0,144).
- **Pengujian di luar skenario terkontrol:** objek non-makanan berhasil diabaikan model dengan baik, tetapi makanan siap saji (nasi putih) salah terdeteksi sebagai tahu/tempe, dan bahan di luar 10 kelas (edamame) salah diklasifikasi sebagai bayam dengan *confidence* tinggi (0,99) — mengindikasikan model belum memiliki mekanisme *reject option* untuk objek *out-of-distribution*.

## Dashboard

Aplikasi web **NutriScan** dibangun dengan *framework* Streamlit dan terdiri dari 4 halaman:

- **Dashboard** — statistik scan harian (jumlah scan, total kalori, bahan terdeteksi) dan riwayat scan terakhir.
- **Scan & Deteksi** — dua mode input (kamera langsung / upload gambar), menampilkan bounding box hasil deteksi beserta confidence, serta input berat manual per bahan (gram).
- **Hasil Estimasi** — tabel estimasi kalori per bahan + total, progress bar terhadap target kalori harian (2.000 kkal), dan ringkasan nutrisi (protein, karbohidrat, lemak, serat) berbasis TKPI.
- **Informasi** — profil pembuat dan detail teknis aplikasi.

Dijalankan secara lokal (`localhost`) sebagai purwarupa untuk memvalidasi model dalam skenario penggunaan nyata.

## Teknologi yang Digunakan

- **Bahasa:** Python 3.x
- **Deep Learning:** PyTorch, torchvision (`ssdlite320_mobilenet_v3_large`), pycocotools
- **Web App:** Streamlit
- **Data & Visualisasi:** pandas, numpy, matplotlib, openpyxl
- **Image Processing:** Pillow (PIL)
- **Anotasi Dataset:** Roboflow (ekspor COCO JSON, auto-orient, resize, augmentasi)
- **Tools:** Visual Studio Code, GPU NVIDIA GeForce RTX 3050 4GB, Microsoft Office (laporan)

## Struktur Proyek

```
SSDLite-FoodDetection-CalorieEstimation/
├── README.md
├── requirements.txt
│
├── Training/
│ ├── Training_SSD_3Fase.py 
│ └── Evaluasi_SSD_3Fase.py 
│
└── Deploy
  ├── model/
  │ └── best_final.pt
  │
  ├── data/
  │ ├── tkpi.xlsx
  │ └── riwayat.csv
  │
  ├── assets/
  │ ├── style.css 
  │ ├── foto/
  │ └── icons/ 
  │
  ├── pages/
  │ ├── 1_scan.py 
  │ ├── 2_hasil.py 
  │ └── 3_informasi.py 
  │
  ├── app.py 
  ├── calorie.py
  ├── detector.py 
  ├── sidebar_component.py 
  └── toggle_component.py 

```

## Cara Menjalankan

1. Clone repository:
```bash
   git clone https://github.com/username/NutriScan.git
   cd NutriScan
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Pastikan bobot model `best_final.pt` sudah tersedia di folder `model/`.
4. Jalankan aplikasi:
```bash
   streamlit run app.py
```
5. *(Opsional)* Melatih ulang model dari awal:
```bash
   python training/Training_SSD_3Fase.py
   python training/Evaluasi_SSD_3Fase.py
```

## Batasan

- Jumlah data pelatihan relatif terbatas (±500 citra/kelas latih, ±50 citra/kelas uji) — kecil untuk standar *deep learning* deteksi objek.
- Evaluasi hanya menggunakan satu partisi data tetap, belum menerapkan *K-Fold Cross-Validation*.
- Augmentasi bersifat *offline*/statis (via Roboflow), belum menggunakan *online augmentation*.
- Estimasi berat bahan makanan bersifat **semi-manual** — pengguna tetap harus menimbang dan menginput berat secara manual; sistem tidak mengestimasi berat dari citra 2D.
- Model rentan terhadap objek *out-of-distribution* (belum ada mekanisme *reject option*) — contoh: nasi putih salah terdeteksi sebagai tahu/tempe, edamame salah terdeteksi sebagai bayam.
- Cakupan kelas masih terbatas pada 10 bahan makanan mentah; belum mencakup ikan, beras, kacang-kacangan, umbi-umbian, dll.
- Perhitungan kalori hanya berdasarkan jenis dan berat bahan, tanpa mempertimbangkan metode pengolahan (digoreng, direbus, dipanggang) yang dapat memengaruhi nilai kalori aktual.
- Sistem tidak mencakup fitur lanjutan seperti rekomendasi diet, akun pengguna, atau integrasi perangkat lain.

## Catatan Privasi dan Etika Data

Repository ini tidak menyertakan kredensial atau API key apa pun. Dataset citra berasal dari kombinasi dataset publik Roboflow dan foto mandiri, tidak mengandung data pribadi maupun wajah individu yang teridentifikasi. Data kalori bersumber dari Tabel Komposisi Pangan Indonesia (TKPI) yang diterbitkan resmi oleh Kementerian Kesehatan RI dan digunakan untuk keperluan riset akademik nonkomersial. Riwayat scan pengguna (`riwayat.csv`) disimpan secara lokal di perangkat dan tidak dikirim atau dibagikan ke pihak ketiga. Aplikasi ini merupakan purwarupa penelitian yang dijalankan secara lokal (`localhost`) untuk validasi model, dan belum melalui audit keamanan untuk deployment produksi/publik.

## Penulis

Rigel Cahyo Gumilang Susanto, Program Studi Sains Data, Universitas Teknologi Yogyakarta. Project ini merupakan Tugas Akhir (skripsi) individu.
