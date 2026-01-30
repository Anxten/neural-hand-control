import cv2
import os
import time

# --- 1. KONFIGURASI ---
DATA_DIR = "data/raw"
CATEGORIES = ['rock', 'paper', 'scissors', 'none']
BOX_SIZE = 250

cap = cv2.VideoCapture(0)

# --- 2. BUAT FOLDER ---
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    if not os.path.exists(path):
        os.makedirs(path)

print("------------------------------------------------")
print("Panduan Kontrol:")
print("Tekan 'r' : Simpan data ROCK (Batu)")
print("Tekan 'p' : Simpan data PAPER (Kertas)")
print("Tekan 's' : Simpan data SCISSORS (Gunting)")
print("Tekan 'n' : Simpan data NONE (Kosong)")
print("Tekan 'q' : Keluar")
print("------------------------------------------------")

count = 0
mode = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Gambar kotak ROI
    h, w, c = frame.shape
    x1, y1 = int(w/2) - 150, 100
    x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
    
    roi = frame[y1:y2, x1:x2]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f"Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if mode != "":
        cv2.putText(frame, f"Recording: {mode.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        filename = f"{time.time()}.jpg"
        path = os.path.join(DATA_DIR, mode, filename)
        cv2.imwrite(path, roi)
        count += 1
        
        if count >= 500:
            mode = ""
            count = 0
            print("Selesai mengambil 500 sampel!")

    cv2.imshow("Data Collection - Tekan r/p/s/n", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        mode = "rock"
        count = 0
        print("Mulai merekam ROCK...")
    elif key == ord('p'):
        mode = "paper"
        count = 0
        print("Mulai merekam PAPER...")
    elif key == ord('s'):
        mode = "scissors"
        count = 0
        print("Mulai merekam SCISSORS...")
    elif key == ord('n'):
        mode = "none"
        count = 0
        print("Mulai merekam NONE...")
    elif key == 32:
        mode = ""
        print("Recording Paused.")

cap.release()
cv2.destroyAllWindows()