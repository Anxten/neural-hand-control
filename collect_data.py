import cv2
import os
import time

# --- KONFIGURASI ---
DATA_DIR = "data/raw"
CATEGORIES = ['rock', 'paper', 'scissors', 'none']
BOX_SIZE = 250  # Ukuran kotak ROI (250x250 pixel)

# Buka Kamera
cap = cv2.VideoCapture(0)

# Cek Folder
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
mode = ""  # Mode perekaman saat ini

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip kamera biar kayak cermin (opsional, enak buat user)
    frame = cv2.flip(frame, 1)
    
    # Gambar Kotak ROI (Region of Interest)
    # Tangan harus masuk ke kotak ini
    h, w, c = frame.shape
    x1, y1 = int(w/2) - 150, 100  # Posisi kotak
    x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
    
    # Ambil gambar hanya di dalam kotak
    roi = frame[y1:y2, x1:x2]
    
    # Gambar kotak di layar utama (Visual saja)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Tampilkan jumlah data yang sudah diambil
    cv2.putText(frame, f"Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if mode != "":
        cv2.putText(frame, f"Recording: {mode.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Simpan file gambar
        # Nama file unik pakai timestamp biar gak ketimpa
        filename = f"{time.time()}.jpg"
        path = os.path.join(DATA_DIR, mode, filename)
        cv2.imwrite(path, roi)
        count += 1
        
        # Kasih delay dikit biar gak terlalu spam (opsional)
        # time.sleep(0.05) 
        
        # Batasi cuma ambil 500 foto sekali tekan (biar gak kebablasan)
        if count >= 500:
            mode = ""
            count = 0
            print("Selesai mengambil 500 sampel!")

    cv2.imshow("Data Collection - Tekan r/p/s/n", frame)
    
    # Input Keyboard
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
    elif key == 32: # Spasi untuk stop manual
        mode = ""
        print("Recording Paused.")

cap.release()
cv2.destroyAllWindows()