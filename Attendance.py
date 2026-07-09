import os
import csv
import time
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO
from deepface import DeepFace

import config
import utils


def load_last_attendance() -> dict:
    """
    Baca log absensi yang sudah ada, kembalikan dict {nama: datetime_terakhir_absen}.
    Dipakai untuk menerapkan cooldown & aturan 1x per hari sejak awal program dijalankan
    (bukan cuma selama sesi berjalan).
    """
    last_seen = {}
    if not os.path.exists(config.ATTENDANCE_LOG_CSV):
        return last_seen

    with open(config.ATTENDANCE_LOG_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            name = row.get("nama")
            if name and (name not in last_seen or ts > last_seen[name]):
                last_seen[name] = ts
    return last_seen


def init_log_file():
    """Buat file CSV log absensi dengan header jika belum ada."""
    utils.ensure_dirs()
    if not os.path.exists(config.ATTENDANCE_LOG_CSV):
        with open(config.ATTENDANCE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nama", "timestamp", "tanggal", "jam"])


def can_log_attendance(name: str, last_seen: dict) -> bool:
    """
    Tentukan apakah orang ini boleh dicatat absen sekarang,
    berdasarkan aturan cooldown & one-absen-per-hari di config.py.
    """
    if name not in last_seen:
        return True

    last_time = last_seen[name]
    now = datetime.now()

    if config.ONE_ABSEN_PER_DAY and last_time.date() == now.date():
        return False

    if now - last_time < timedelta(minutes=config.COOLDOWN_MINUTES):
        return False

    return True


def log_attendance(name: str):
    """Tulis satu baris record absensi ke CSV."""
    now = datetime.now()
    with open(config.ATTENDANCE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
        ])
    print(f"✅ Absen tercatat: {name} pada {now.strftime('%Y-%m-%d %H:%M:%S')}")


def recognize_face(face_crop_path: str):
    """
    Cocokkan satu crop wajah ke database menggunakan DeepFace.
    Return: (nama_dikenali atau None, jarak_kemiripan atau None)
    """
    try:
        dfs = DeepFace.find(
            img_path=face_crop_path,
            db_path=config.DATASET_DIR,
            model_name=config.DEEPFACE_MODEL_NAME,
            distance_metric=config.DEEPFACE_DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return None, None

    if len(dfs) == 0 or dfs[0].empty:
        return None, None

    best_match = dfs[0].iloc[0]
    distance_col = [c for c in best_match.index if "distance" in c.lower()]
    distance = best_match[distance_col[0]] if distance_col else None

    if distance is not None and distance > config.DEEPFACE_THRESHOLD:
        return None, distance  # terlalu jauh, anggap tidak dikenal

    identity_path = best_match["identity"]
    # identity_path formatnya: dataset_wajah/<nama>/<file>.jpg -> ambil nama foldernya
    name = os.path.basename(os.path.dirname(identity_path))
    return name, distance


def run_attendance():
    utils.ensure_dirs()
    init_log_file()

    if not os.listdir(config.DATASET_DIR):
        print("⚠️  Database wajah (dataset_wajah/) masih kosong.")
        print("Jalankan register.py terlebih dahulu untuk mendaftarkan wajah.")
        return

    model_path = utils.ensure_yolo_model()
    model_yolo = YOLO(model_path)

    last_seen = load_last_attendance()

    print("Menyiapkan database wajah (precompute embedding)...")
    # Memicu DeepFace membangun cache embedding sekali di awal, bukan saat frame pertama live,
    # supaya tidak ada jeda/lag mendadak saat sistem sudah berjalan.
    dummy_files = []
    for person in os.listdir(config.DATASET_DIR):
        person_dir = os.path.join(config.DATASET_DIR, person)
        if os.path.isdir(person_dir):
            imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))]
            if imgs:
                dummy_files.append(os.path.join(person_dir, imgs[0]))
                break
    if dummy_files:
        try:
            DeepFace.find(
                img_path=dummy_files[0],
                db_path=config.DATASET_DIR,
                model_name=config.DEEPFACE_MODEL_NAME,
                distance_metric=config.DEEPFACE_DISTANCE_METRIC,
                enforce_detection=False,
                silent=True,
            )
            print("Database siap.\n")
        except Exception as e:
            print(f"Peringatan saat precompute: {e}")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Gagal membuka kamera. Periksa CAMERA_INDEX di config.py.")
        return

    print("Sistem absensi berjalan. Tunjukkan wajah ke kamera. Tekan 'q' untuk keluar.\n")

    temp_crop_path = os.path.join(config.BASE_DIR if hasattr(config, "BASE_DIR") else ".", "temp_face.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        results = model_yolo(frame, verbose=False, conf=config.YOLO_CONFIDENCE)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            label = "Tidak Dikenal"
            color = config.BOX_COLOR_UNKNOWN

            if face_crop.size != 0:
                cv2.imwrite(temp_crop_path, face_crop)
                name, distance = recognize_face(temp_crop_path)

                if name:
                    color = config.BOX_COLOR_KNOWN
                    if can_log_attendance(name, last_seen):
                        log_attendance(name)
                        last_seen[name] = datetime.now()
                        label = f"{name} (Absen OK)"
                    else:
                        label = f"{name} (Sudah absen)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE, color, 2)

        cv2.imshow("Sistem Absensi Wajah - tekan q untuk keluar", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_crop_path):
        os.remove(temp_crop_path)
    print("\nSistem absensi dihentikan.")


if __name__ == "__main__":
    run_attendance()