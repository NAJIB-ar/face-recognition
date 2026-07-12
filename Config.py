import os

# ── Path & Direktori ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_wajah")   # database wajah terdaftar
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_LOG_CSV = os.path.join(LOG_DIR, "absensi.csv")

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov11n-face.pt")
YOLO_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"

FACE_LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ── Kamera ────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = kamera default/webcam laptop.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ── Registrasi Wajah ─────────────────────────────────────────────
PHOTOS_PER_PERSON = 5     # jumlah foto yang diambil saat registrasi (variasi sudut/ekspresi)
CAPTURE_DELAY_SEC = 1.0   # jeda antar foto saat registrasi (beri waktu ubah pose)

# ── Deteksi Wajah (YOLO) ──────────────────────────────────────────
YOLO_CONFIDENCE = 0.25   # diturunkan jika wajah susah terdeteksi atau naik jika banyak deteksi salah

# ── Pengenalan Wajah (DeepFace) ───────────────────────────────────
DEEPFACE_MODEL_NAME = "Facenet"     # model embedding wajah
DEEPFACE_DISTANCE_METRIC = "cosine" # metrik jarak untuk pencocokan
DEEPFACE_THRESHOLD = 0.40           # ambang batas kemiripan (semakin kecil = semakin ketat)

# ── Logika Absensi ────────────────────────────────────────────────
COOLDOWN_MINUTES = 60      # jarak minimum antar absen untuk orang yang sama (menit)
ONE_ABSEN_PER_DAY = True   # jika True, hanya izinkan 1x absen sukses per orang per hari
EAR_THRESHOLD = 0.20       # Ambang batas kedipan mata (Liveness Detection). Nilai di bawah ini = mata tertutup.

# ── Tampilan ───────────────────────────────────────────────────────
BOX_COLOR_KNOWN = (0, 255, 0)      # hijau (BGR)
BOX_COLOR_UNKNOWN = (0, 0, 255)    # merah (BGR)
FONT_SCALE = 0.7