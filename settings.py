# settings.py

# =================== Konfigurasi Kamera & ROI ===================
# Koordinat titik untuk transformasi perspektif (TopLeft, TopRight, BottomRight, BottomLeft)
# Sesuaikan dengan setup kamera Anda
PTS1_COORDINATES = [[1339, 105], [2255, 135], [2622, 847], [1155, 880]]

# Dimensi nyata dari area yang dicakup oleh PTS1_COORDINATES
REAL_HEIGHT_M = 6  # Tinggi/panjang area dalam meter
REAL_WIDTH_M = 6    # Lebar area dalam meter

# =================== Konfigurasi Video/Stream ===================
# Path ke file video atau URL RTSP stream
# Contoh RTSP: "rtsp://username:password@ip_address:port/stream_path"
# Contoh File: "/path/to/your/video.mp4" atau "C:/videos/my_video.avi"
# VIDEO_PATH = "rtsp://admin:081Sultan@10.5.79.69:554/Streaming/Channels/1"
# VIDEO_PATH = "path/to/your/local_video.mp4" # Alternatif untuk file lokal
VIDEO_PATH ="output_20km_4.2_3.mp4"

# Nama file untuk output video (jika memproses file video, bukan stream)
OUTPUT_VIDEO_FILE = "output_run_real.mp4"

# Durasi pemrosesan video dalam detik (opsional, bisa di-override atau tidak digunakan jika stream)
# Jika None, proses seluruh video file atau stream tanpa batas waktu tertentu dari sini
PROCESSING_DURATION_SECONDS = 600

# Interval skip frame (1 berarti proses setiap frame, 2 berarti proses setiap frame kedua, dst.)
SKIP_FRAME_INTERVAL = 1

# Ukuran buffer untuk OpenCV VideoCapture (terutama untuk RTSP)
CV2_BUFFER_SIZE = 20

# FPS video (jika diketahui dan konstan, jika tidak SpeedEstimator akan mencoba menghitungnya)
# Untuk RTSP, ini mungkin lebih sebagai target atau referensi.
VIDEO_FPS = 30

# =================== Konfigurasi Model YOLO ===================
# Path ke file model YOLO (.pt)
YOLO_MODEL_PATH = "./yolov8x.pt"

# Apakah akan menggunakan CUDA untuk inferensi jika tersedia
USE_CUDA = True

# Apakah akan melakukan preload/warmup model saat startup
PRELOAD_MODEL = True

# Confidence threshold minimum untuk deteksi YOLO
MIN_DETECTION_CONFIDENCE = 0.4

# Kelas yang akan dideteksi (misal: 2 untuk mobil, 3 untuk motor di COCO)
# Biarkan None atau [] untuk mendeteksi semua kelas yang dilatih model
YOLO_CLASSES_TO_DETECT = [2, 3] # Contoh: mobil dan motor

# =================== Konfigurasi Estimasi Kecepatan ===================
# Batas kecepatan maksimum dalam km/jam. Pelanggaran akan dicatat di atas nilai ini.
SPEED_THRESHOLD_KMH = 38.0

# Arah lalu lintas yang diizinkan. Pilihan: "top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"
ALLOWED_DIRECTION = "top_to_bottom"

# Piksel per meter untuk kalibrasi (dapat disesuaikan berdasarkan setup)
PIXELS_PER_METER = 100

# Batas kecepatan tertinggi yang dianggap masuk akal sebelum capping agresif (untuk menangani outlier)
OUTLIER_SPEED_THRESHOLD_KMH = 80.0

# Konfigurasi untuk tracker (ByteTracker)
TRACKER_ARGS = {
    "track_thresh": 0.25,    # Ambang batas deteksi untuk memulai track baru
    "track_buffer": 30,      # Jumlah frame untuk menyimpan track yang hilang sebelum dihapus
    "match_thresh": 0.8,     # Ambang batas matching untuk asosiasi ulang track
    "mot20": False,          # Apakah menggunakan format MOT20 (biasanya False)
    "fuse_score": True       # Apakah menggabungkan skor deteksi (rekomendasi ByteTrack)
}

# Aturan kalibrasi kecepatan: (min_speed, max_speed, factor_koreksi)
# Digunakan untuk menyesuaikan kecepatan yang dihitung berdasarkan rentang kecepatan
CALIBRATION_RULES = [
    (1, 10, 0.95),
    (10, 20, 0.92),
    (20, 30, 0.8),
    (30, 40, 0.8),
    (40, OUTLIER_SPEED_THRESHOLD_KMH + 20, 0.85),
]

# =================== Konfigurasi RabbitMQ ===================
USE_RABBITMQ = True # Set ke False untuk menonaktifkan integrasi RabbitMQ
RABBITMQ_HOST = "10.15.40.194"
RABBITMQ_PORT = 5672
RABBITMQ_VHOST = "/"
RABBITMQ_USER = "admin"
RABBITMQ_PASS = "admin"
RABBITMQ_QUEUE_NAME = "whatsapp_messages"
RABBITMQ_HEARTBEAT = 60
RABBITMQ_BLOCKED_CONNECTION_TIMEOUT = 300

# Nomor WA atau ID Grup WA untuk notifikasi (sesuaikan)
# Untuk grup: dapatkan dari "ID Grup" setelah mengirim pesan via API atau inspect di WhatsApp Web
# Contoh nomor pribadi: "6281234567890"
# Contoh ID grup: "120363021234567890@g.us"
WA_RECIPIENT_NUMBER = "120363418805481164@g.us" # ID Grup contoh
# WA_RECIPIENT_NUMBER = "6289523804019" # Nomor pribadi contoh

# =================== Konfigurasi Pemrosesan & Monitoring ===================
# Ukuran maksimum antrian frame (untuk RTSPStreamManager dan FrameProcessor)
MAX_QUEUE_SIZE = 60

# Jumlah thread untuk pemrosesan frame (jika menggunakan FrameProcessor)
# Jika 0 atau 1, pemrosesan dilakukan di thread utama.
# Nilai ini ada di skrip Anda sebagai `processing_threads` yang diinisialisasi ke 16
# dan kemudian digunakan di `FrameProcessor` jika diaktifkan.
# Untuk saat ini, jika Anda tidak menggunakan `FrameProcessor` secara aktif, ini mungkin tidak relevan.
PROCESSING_THREADS = 1 # Default ke 1 jika tidak pakai FrameProcessor, sesuaikan jika pakai

# Interval monitoring sumber daya dalam detik
MONITORING_INTERVAL_SECONDS = 1.0
# File log untuk monitoring sumber daya
RESOURCE_MONITORING_LOG_FILE = "resource_usage_n.log"

# =================== Konfigurasi Lainnya ===================
# Mode debug untuk SpeedEstimator (akan mencetak lebih banyak log)
DEBUG_MODE = False

# Folder untuk menyimpan gambar pelanggar
SPEED_VIOLATORS_DIR = "speed_violators"
COUNTERFLOW_VIOLATORS_DIR = "counterflow_violators"

# Kualitas gambar JPEG untuk pesan base64 (0-100, lebih rendah = ukuran lebih kecil)
JPEG_QUALITY_FOR_BASE64 = 65

# Faktor ekspansi BBox untuk gambar pelanggaran
VIOLATION_IMAGE_BBOX_EXPANSION_FACTOR = 4

# Panjang buffer untuk cek konsistensi kecepatan (dalam detik)
# Pelanggaran kecepatan hanya dikirim jika kecepatan di atas threshold selama durasi ini.
# Jika 0, setiap frame di atas threshold akan dianggap pelanggaran (setelah EMA smoothing)
CONSISTENT_SPEED_BUFFER_SECONDS = 0.5

# Minimum frame dalam buffer konsistensi (override jika CONSISTENT_SPEED_BUFFER_SECONDS * VIDEO_FPS < nilai ini)
MIN_CONSISTENT_SPEED_FRAMES = 3


# --- Helper untuk memuat TrackerArgs dari settings ---
class TrackerArgsFromSettings:
    def __init__(self):
        self.track_thresh = TRACKER_ARGS["track_thresh"]
        self.track_buffer = TRACKER_ARGS["track_buffer"]
        self.match_thresh = TRACKER_ARGS["match_thresh"]
        self.mot20 = TRACKER_ARGS["mot20"]
        self.fuse_score = TRACKER_ARGS["fuse_score"]