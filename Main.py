from ultralytics import YOLO

import utils
from register import register_new_face
from attendance import run_attendance

def main():
    utils.ensure_dirs()
    model_path = utils.ensure_yolo_model()
    
    
    print("="*50)
    print(" Sistem Absensi Wajah (YOLOv11n-face + DeepFace) ")
    print("="*50)
    
    while True:
        print("\nPilih menu:")
        print("  1. Registrasi wajah baru")
        print("  2. Mulai sesi absensi (live)")
        print("  3. Keluar")
        choice = input("Masukkan pilihan (1/2/3): ").strip()
 
        if choice == "1":
            model_yolo = YOLO(model_path)
            register_new_face(model_yolo)
        elif choice == "2":
            run_attendance()
        elif choice == "3":
            print("Program selesai.")
            break
        else:
            print("Pilihan tidak valid, coba lagi.")
 
 
if __name__ == "__main__":
    main()