import os
import re
import urllib.request
import math
import config


def sanitize_name(raw_name: str) -> str:
    # cleaning nama
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)                 # spasi -> underscore
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)       # buang karakter selain huruf/angka/_/-
    return name


def ensure_dirs():
    # Pastikan semua direktori yang dibutuhkan sistem sudah ada.
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)


def ensure_yolo_model():
    """
    Pastikan file model YOLO face tersedia di models/.
    Jika belum ada, download otomatis dari sumber di config.py.
    """
    if os.path.exists(config.YOLO_MODEL_PATH):
        return config.YOLO_MODEL_PATH

    print(f"Model YOLO tidak ditemukan. Mengunduh dari {config.YOLO_MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(config.YOLO_MODEL_URL, config.YOLO_MODEL_PATH)
        print(f"Model berhasil diunduh ke {config.YOLO_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Gagal mengunduh model YOLO secara otomatis: {e}\n"
            f"Silakan unduh manual dari {config.YOLO_MODEL_URL} "
            f"dan simpan ke {config.YOLO_MODEL_PATH}"
        )
    return config.YOLO_MODEL_PATH


def ensure_face_landmarker_model():
    if os.path.exists(config.FACE_LANDMARKER_MODEL_PATH):
        return config.FACE_LANDMARKER_MODEL_PATH

    print(f"Model Face Landmarker tidak ditemukan. Mengunduh dari {config.FACE_LANDMARKER_MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(config.FACE_LANDMARKER_MODEL_URL, config.FACE_LANDMARKER_MODEL_PATH)
        print(f"Model berhasil diunduh ke {config.FACE_LANDMARKER_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Gagal mengunduh model Face Landmarker secara otomatis: {e}\n"
            f"Silakan unduh manual dari {config.FACE_LANDMARKER_MODEL_URL} "
            f"dan simpan ke {config.FACE_LANDMARKER_MODEL_PATH}"
        )
    return config.FACE_LANDMARKER_MODEL_PATH


def person_exists(name: str) -> bool:
    """Cek apakah nama (setelah sanitasi) sudah terdaftar di database wajah."""
    person_dir = os.path.join(config.DATASET_DIR, name)
    return os.path.isdir(person_dir) and len(os.listdir(person_dir)) > 0


def clear_deepface_cache():
    # Hapus semua file cache embedding (.pkl)
    if not os.path.isdir(config.DATASET_DIR):
        return
    for fname in os.listdir(config.DATASET_DIR):
        if fname.endswith(".pkl"):
            try:
                os.remove(os.path.join(config.DATASET_DIR, fname))
                print(f"Cache embedding lama dihapus: {fname}")
            except OSError:
                pass

# Definisi urutan titik landmark kelopak mata berdasarkan MediaPipe Face Mesh
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def get_ear_from_landmarks(landmarks, width, height):
    # Menghitung rata-rata Eye Aspect Ratio (EAR) dari kedua mata.
    def dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        
    def calculate_ear(eye_indices):
        # Ambil koordinat pixel (x, y) dari landmark MediaPipe
        pts = [(landmarks[idx].x * width, landmarks[idx].y * height) for idx in eye_indices]
        
        # Jarak vertikal kelopak mata
        v1 = dist(pts[1], pts[5])
        v2 = dist(pts[2], pts[4])
        
        # Jarak horizontal ujung mata
        h = dist(pts[0], pts[3])
        
        if h == 0:
            return 0
        return (v1 + v2) / (2.0 * h)
        
    right_ear = calculate_ear(RIGHT_EYE_INDICES)
    left_ear = calculate_ear(LEFT_EYE_INDICES)
    
    return (right_ear + left_ear) / 2.0