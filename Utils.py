"""
utils.py
Fungsi-fungsi bantu yang dipakai di beberapa modul.
"""

import os
import re
import urllib.request
import config


def sanitize_name(raw_name: str) -> str:
    """
    Membersihkan input nama dari user agar aman dipakai sebagai nama file/folder.
    - Spasi -> underscore
    - Hanya izinkan huruf, angka, underscore, strip
    - Buang karakter aneh/simbol

    Contoh: "Budi Santoso!!" -> "Budi_Santoso"
    """
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)                 # spasi -> underscore
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)       # buang karakter selain huruf/angka/_/-
    return name


def ensure_dirs():
    """Pastikan semua direktori yang dibutuhkan sistem sudah ada."""
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


def person_exists(name: str) -> bool:
    """Cek apakah nama (setelah sanitasi) sudah terdaftar di database wajah."""
    person_dir = os.path.join(config.DATASET_DIR, name)
    return os.path.isdir(person_dir) and len(os.listdir(person_dir)) > 0


def clear_deepface_cache():
    """
    Hapus semua file cache embedding (.pkl) yang dibuat DeepFace di dalam DATASET_DIR.
    Wajib dipanggil setelah registrasi wajah baru, agar DeepFace membangun ulang
    representasi embedding dan orang baru langsung bisa dikenali.

    Catatan: format nama file cache DeepFace berbeda-beda tergantung versi library
    (contoh lama: representations_facenet.pkl,
     contoh baru: ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl),
    jadi di sini kita hapus SEMUA file .pkl di dalam DATASET_DIR, bukan cuma yang match satu pola nama.
    """
    if not os.path.isdir(config.DATASET_DIR):
        return
    for fname in os.listdir(config.DATASET_DIR):
        if fname.endswith(".pkl"):
            try:
                os.remove(os.path.join(config.DATASET_DIR, fname))
                print(f"Cache embedding lama dihapus: {fname}")
            except OSError:
                pass