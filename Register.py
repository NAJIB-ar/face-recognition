import os
import time
import cv2
from ultralytics import YOLO

import config
import utils


def register_new_face(model_yolo: YOLO):
    utils.ensure_dirs()

    raw_name = input("Masukkan nama untuk registrasi: ").strip()
    if not raw_name:
        print("Nama tidak boleh kosong. Registrasi dibatalkan.")
        return

    name = utils.sanitize_name(raw_name)
    if not name:
        print("Nama tidak valid setelah dibersihkan. Gunakan huruf/angka. Registrasi dibatalkan.")
        return

    if utils.person_exists(name):
        confirm = input(
            f"Nama '{name}' sudah terdaftar. Tambah foto baru ke akun yang sama? (y/n): "
        ).strip().lower()
        if confirm != "y":
            print("Registrasi dibatalkan.")
            return

    person_dir = os.path.join(config.DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing_count = len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))])

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Gagal membuka kamera. Periksa CAMERA_INDEX di config.py.")
        return

    print(f"\nMemulai registrasi untuk '{name}'.")
    print(f"Sistem akan mengambil {config.PHOTOS_PER_PERSON} foto otomatis.")
    print("Ubah sedikit posisi wajah (miring kiri/kanan, sedikit jauh/dekat) tiap foto diambil.")
    print("Tekan 'q' kapan saja untuk membatalkan.\n")

    photos_taken = 0
    last_capture_time = 0

    while photos_taken < config.PHOTOS_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        results = model_yolo(frame, verbose=False, conf=config.YOLO_CONFIDENCE)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        display_frame = frame.copy()
        face_ready = False
        best_box = None

        if len(boxes) == 1:
            # Hanya proses jika PERSIS satu wajah terdeteksi (hindari salah simpan wajah orang lain)
            x1, y1, x2, y2 = map(int, boxes[0][:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            best_box = (x1, y1, x2, y2)
            face_ready = True
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif len(boxes) > 1:
            cv2.putText(display_frame, "Terdeteksi > 1 wajah, pastikan hanya 1 orang di kamera",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Wajah tidak terdeteksi",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Foto: {photos_taken}/{config.PHOTOS_PER_PERSON}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Registrasi Wajah - tekan q untuk batal", display_frame)

        now = time.time()
        if face_ready and (now - last_capture_time) >= config.CAPTURE_DELAY_SEC:
            x1, y1, x2, y2 = best_box
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size != 0:
                idx = existing_count + photos_taken + 1
                filename = os.path.join(person_dir, f"{name}_{idx}.jpg")
                cv2.imwrite(filename, face_crop)
                photos_taken += 1
                last_capture_time = now
                print(f"  Foto {photos_taken}/{config.PHOTOS_PER_PERSON} tersimpan -> {filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registrasi dibatalkan oleh user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if photos_taken > 0:
        utils.clear_deepface_cache()
        print(f"\n✅ Registrasi selesai. {photos_taken} foto baru disimpan untuk '{name}'.")
    else:
        print("\nTidak ada foto yang tersimpan.")


if __name__ == "__main__":
    utils.ensure_dirs()
    model_path = utils.ensure_yolo_model()
    model_yolo = YOLO(model_path)
    register_new_face(model_yolo)