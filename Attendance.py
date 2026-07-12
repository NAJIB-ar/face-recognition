import os
import csv
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace

import config
import utils


def load_last_attendance() -> dict:
    """
    Baca log absensi yang sudah ada, kembalikan dict {nama: datetime_terakhir_absen}.
    Dipakai untuk menerapkan cooldown & aturan 1x per hari sejak program dijalankan
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
    # create file CSV log absensi
    utils.ensure_dirs()
    if not os.path.exists(config.ATTENDANCE_LOG_CSV):
        with open(config.ATTENDANCE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nama", "timestamp", "tanggal", "jam"])


def can_log_attendance(name: str, last_seen: dict) -> bool:
    # aturan cooldown & one-absen-per-hari di config.py.
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
    # Tulis satu baris record absensi ke CSV
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
    # Cocokkan wajah ke database menggunakan DeepFace.
    try:
        dfs = DeepFace.find(
            img_path=face_crop_path,
            db_path=config.DATASET_DIR,
            model_name=config.DEEPFACE_MODEL_NAME,
            distance_metric=config.DEEPFACE_DISTANCE_METRIC,
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return None, None

    if len(dfs) == 0 or dfs[0].empty:
        print("[DEBUG] Tidak ada kandidat sama sekali dari DeepFace.find (database kosong atau tidak ada hasil).")
        return None, None

    best_match = dfs[0].iloc[0]
    distance_col = [c for c in best_match.index if "distance" in c.lower()]
    distance = best_match[distance_col[0]] if distance_col else None

    identity_path = best_match["identity"]
    candidate_name = os.path.basename(os.path.dirname(identity_path))
    print(f"[DEBUG] Kandidat terdekat: {candidate_name} | distance: {distance} | threshold: {config.DEEPFACE_THRESHOLD}")

    if distance is not None and distance > config.DEEPFACE_THRESHOLD:
        return None, distance  # terlalu jauh, anggap tidak dikenal

    name = candidate_name
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
                detector_backend="skip",
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

    # Inisialisasi model MediaPipe Face Landmarker (Tasks API) untuk Liveness Detection
    landmarker_model_path = utils.ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=10)
    
    face_landmarker = FaceLandmarker.create_from_options(options)

    
    # State melacak status kedipan tiap orang: {nama: {"eye_closed": False, "blinked": False}}
    blink_states = {}

    print("Sistem absensi berjalan. Tunjukkan wajah ke kamera. Tekan 'q' untuk keluar.\n")

    temp_crop_path = os.path.join(config.BASE_DIR if hasattr(config, "BASE_DIR") else ".", "temp_face.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # =========================================================================
        # PROSES MEDIAPIPE FACE MESH (Untuk mencari landmark kelopak mata)
        # =========================================================================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        mesh_results = face_landmarker.detect(mp_image)
        
        # Simpan nilai EAR untuk setiap wajah yang terdeteksi oleh MediaPipe
        mesh_ears = []
        if mesh_results.face_landmarks:
            for landmarks in mesh_results.face_landmarks:
                ear = utils.get_ear_from_landmarks(landmarks, frame.shape[1], frame.shape[0])
                
                # Cari pusat koordinat wajah dari MediaPipe (untuk dicocokkan dengan YOLO nanti)
                x_coords = [lm.x * frame.shape[1] for lm in landmarks]
                y_coords = [lm.y * frame.shape[0] for lm in landmarks]

                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                mesh_ears.append({"center": (center_x, center_y), "ear": ear})

        # =========================================================================
        # PROSES YOLO FACE DETECTION
        # =========================================================================
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
                
                # Cari nilai EAR yang cocok untuk wajah ini (berdasarkan posisi tengah/center wajah)
                current_ear = None
                for mesh in mesh_ears:
                    # Jika pusat wajah MediaPipe berada di dalam area kotak YOLO ini
                    if x1 <= mesh["center"][0] <= x2 and y1 <= mesh["center"][1] <= y2:
                        current_ear = mesh["ear"]
                        break

                if name:
                    if can_log_attendance(name, last_seen):
                        # Inisialisasi state kedipan orang ini jika belum ada
                        if name not in blink_states:
                            blink_states[name] = {"eye_closed": False, "blinked": False}
                            
                        # Logika mendeteksi kedipan mata (EAR drop)
                        if current_ear is not None:
                            if current_ear < config.EAR_THRESHOLD:
                                blink_states[name]["eye_closed"] = True
                            elif current_ear > config.EAR_THRESHOLD and blink_states[name]["eye_closed"]:
                                blink_states[name]["blinked"] = True
                                blink_states[name]["eye_closed"] = False
                                
                        if blink_states[name]["blinked"]:
                            # Syarat terpenuhi: Wajah asli + Berkedip -> Catat absen
                            log_attendance(name)
                            last_seen[name] = datetime.now()
                            label = f"{name} (Absen OK)"
                            color = config.BOX_COLOR_KNOWN
                            
                            # Reset status agar siap untuk sesi berikutnya
                            blink_states[name]["blinked"] = False 
                        else:
                            # Minta user untuk berkedip sebagai bukti liveness
                            label = f"{name} (Berkedip utk Absen)"
                            color = (0, 255, 255) # Warna kuning (BGR) untuk peringatan / intruksi
                    else:
                        color = config.BOX_COLOR_KNOWN
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