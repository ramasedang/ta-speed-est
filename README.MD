# Deteksi Kecepatan & Pelanggaran Arus Lalu Lintas 🚗💨

Proyek ini mendeteksi kecepatan kendaraan dan pelanggaran melawan arus dari stream video atau file video menggunakan YOLO dan ByteTrack. Notifikasi pelanggaran dapat dikirim melalui RabbitMQ.

---
## 🚦 Persiapan & Instalasi

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek:

### 1. Instalasi `uv` (Python Package Installer)

`uv` digunakan untuk manajemen dependensi yang cepat.

* **macOS dan Linux:**
    Gunakan `curl` (atau `wget`) untuk mengunduh dan menjalankan skrip instalasi:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Atau jika menggunakan `wget`:
    ```bash
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```


* **Windows:**
    Kunjungi [dokumentasi resmi `uv`](https://astral.sh/uv#installation) untuk instruksi instalasi di Windows.

---
### 2. Dapatkan Kode Proyek

Kloning repositori ini ke mesin lokal Anda. Ganti `[URL_REPO_ANDA]` dengan URL repositori Git Anda:

```bash
git clone [URL_REPO_ANDA]
cd ta-speed-est # atau nama direktori proyek Anda
```
Jika Anda tidak menggunakan Git, pastikan semua file proyek (`speed_est.py`, `settings.py`, `pyproject.toml`, `docker-compose.yaml`, dll.) sudah ada di direktori kerja Anda.

---
### 3. Instalasi Dependensi menggunakan `uv`

Setelah `uv` terinstal dan kode proyek sudah ada, instal semua dependensi yang dibutuhkan. Proyek ini menggunakan `pyproject.toml` untuk manajemen dependensi dengan `uv`.

1.  **(Opsional, tapi direkomendasikan) Buat dan Aktifkan Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # Untuk Linux/macOS
    # .\.venv\Scripts\activate # Untuk Windows PowerShell
    ```

2.  **Sinkronkan/Instal Dependensi:**
    Jalankan perintah berikut di direktori root proyek (`~/SSUDOSSD/project/speed-est/`):
    ```bash
    uv sync
    ```
    Ini akan menginstal semua paket yang tercantum di `pyproject.toml` dan `uv.lock`.

---
## 🛠️ Konfigurasi Aplikasi (`settings.py`)

Sebelum menjalankan aplikasi, **sangat penting** untuk menyesuaikan konfigurasi dalam file `settings.py`. File ini berisi semua parameter krusial.

Buka file `settings.py` dan perbarui nilai-nilai berikut sesuai dengan setup Anda:

* **`PTS1_COORDINATES`**: Koordinat titik-titik pada frame video yang mendefinisikan Region of Interest (ROI) untuk transformasi perspektif. Ini penting untuk kalibrasi pengukuran jarak dan kecepatan.
    * **Cara Mendapatkan Koordinat**:
        1.  Ambil sebuah frame representatif dari video atau stream Anda.
        2.  Unggah frame tersebut ke [**Roboflow Polygon Zone**](https://polygonzone.roboflow.com/).
        3.  Gambarkan poligon (zona) yang mencakup area jalan yang ingin Anda pantau. Pastikan urutan titiknya adalah: **Kiri-Atas, Kanan-Atas, Kanan-Bawah, Kiri-Bawah**.
        4.  Salin koordinat (x, y) untuk setiap titik yang dihasilkan oleh alat tersebut.

        **Contoh Visual Penentuan Zona:**
        <img width="700" alt="Contoh Penentuan Zona ROI" src="https://github.com/user-attachments/assets/ac6939f2-1eaf-4311-9689-8a1069d6d633" />
        *Gambar di atas menunjukkan contoh poligon yang digambar pada frame jalan. Anda akan mendapatkan 4 pasang koordinat (x,y) dari alat tersebut.*

    * Format di `settings.py`: `[[x_kiri_atas, y_kiri_atas], [x_kanan_atas, y_kanan_atas], [x_kanan_bawah, y_kanan_bawah], [x_kiri_bawah, y_kiri_bawah]]`.
* **`REAL_HEIGHT_M`** & **`REAL_WIDTH_M`**: Dimensi area nyata (dalam meter) yang dicakup oleh `PTS1_COORDINATES`. Ukur lebar dan panjang (atau tinggi, tergantung orientasi kamera) sebenarnya dari zona yang Anda gambar di dunia nyata.
* **`VIDEO_PATH`**: Path ke file video lokal atau URL RTSP stream.
* **`YOLO_MODEL_PATH`**: Path ke file model YOLO (`.pt`). Pastikan model sudah diunduh.
* **`SPEED_THRESHOLD_KMH`**: Batas kecepatan maksimum dalam km/jam.
* **Konfigurasi RabbitMQ** (jika `USE_RABBITMQ = True`):
    * `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_VHOST`, `RABBITMQ_USER`, `RABBITMQ_PASS`, `RABBITMQ_QUEUE_NAME`.
    * `WA_RECIPIENT_NUMBER`: Nomor WhatsApp atau ID Grup tujuan notifikasi.
* **Parameter Lain**:
    * `OUTPUT_VIDEO_FILE`: Nama file output jika memproses video lokal.
    * `ALLOWED_DIRECTION`: Arah lalu lintas yang diizinkan.
    * `USE_CUDA`: `True` untuk menggunakan GPU (NVIDIA CUDA).
    * `MIN_DETECTION_CONFIDENCE`: Ambang batas kepercayaan minimum deteksi YOLO.
    * `SPEED_VIOLATORS_DIR` & `COUNTERFLOW_VIOLATORS_DIR`: Direktori untuk menyimpan gambar bukti pelanggaran.

Pastikan semua path file dan URL sudah benar.

---
## 🐳 (Opsional) Menjalankan RabbitMQ dengan Docker

Jika Anda menggunakan RabbitMQ untuk notifikasi, Anda bisa menjalankannya menggunakan Docker Compose dengan file `docker-compose.yaml` yang tersedia.

1.  Pastikan Docker dan Docker Compose sudah terinstal.
2.  Sesuaikan `RABBITMQ_DEFAULT_USER` dan `RABBITMQ_DEFAULT_PASS` di `docker-compose.yaml` agar cocok dengan yang ada di `settings.py`.
3.  Jalankan dari direktori root proyek:
    ```bash
    docker-compose up -d
    ```
    Ini akan menjalankan RabbitMQ di background. Anda bisa mengakses UI manajemennya di `http://localhost:15672`.

---
## ▶️ Menjalankan Aplikasi

Setelah semua dependensi terinstal dan `settings.py` (serta RabbitMQ jika digunakan) telah dikonfigurasi:

1.  Pastikan Anda berada di direktori root proyek (`~/SSUDOSSD/project/speed-est/`).
2.  Jika menggunakan virtual environment, pastikan sudah aktif.
3.  Jalankan skrip utama:
    ```bash
    python speed_est.py
    ```
    atau
    ```
    uv run speed_est.py
    ```

Aplikasi akan mulai memproses video sesuai konfigurasi. Log akan ditampilkan di terminal. Kalau ingin di background bisa pakai supervisor atau nohup
