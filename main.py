import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

# --- 1. KONFIGURASI ---
MODEL_PATH = "models/hand_gesture_cnn.pth"
CLASSES = ['none', 'paper', 'rock', 'scissors'] # Urutan harus sesuai alfabet folder tadi!
BOX_SIZE = 250
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DEFINISI ULANG MODEL (Wajib sama persis dengan train.py) ---
class HandGestureCNN(nn.Module):
    def __init__(self):
        super(HandGestureCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully Connected + DROPOUT
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        
        # DROPOUT: Membuang 50% informasi neuron secara acak.
        # Efek: Mencegah AI "menghafal" posisi tertentu. Dia harus paham bentuk global.
        self.dropout = nn.Dropout(0.5) 
        
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # Terapkan dropout sebelum layer terakhir
        x = self.fc2(x)
        return x

# --- 3. LOAD OTAK AI ---
print("🧠 Memuat model AI...")
model = HandGestureCNN().to(DEVICE)
try:
    # Load weight ke CPU agar aman (map_location)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Mode evaluasi (matikan dropout/batchnorm training)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Transformasi gambar (Sama persis dengan training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# --- 4. MULAI KAMERA ---
cap = cv2.VideoCapture(0)

print("📷 Kamera siap! Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Cermin
    
    # Ambil ROI (Kotak Hijau)
    h, w, c = frame.shape
    x1, y1 = int(w/2) - 150, 100
    x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
    
    # Potong gambar kotak hijau
    roi = frame[y1:y2, x1:x2]
    
    # --- PREDIKSI AI ---
    # 1. Ubah warna BGR (OpenCV) ke RGB (PyTorch)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # 2. Transformasi jadi Tensor
    roi_tensor = transform(roi_rgb).unsqueeze(0).to(DEVICE) # Tambah dimensi batch [1, 3, 64, 64]
    
    # 3. Prediksi
    with torch.no_grad(): # Gak perlu hitung gradien (hemat memori)
        outputs = model(roi_tensor)
        
        # Hitung Probabilitas (Softmax)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Ambil nilai tertinggi (Argmax)
        confidence, predicted = torch.max(probs, 1)
        
        class_name = CLASSES[predicted.item()]
        conf_score = confidence.item() * 100 # Persentase

    # --- VISUALISASI ---
    # Gambar Kotak
    color = (0, 255, 0) # Hijau
    
    # Kalau AI yakin > 80%, tulis nama jurusnya
    label_text = f"{class_name.upper()} ({conf_score:.1f}%)"
    
    # Tampilkan teks
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "Letakkan tangan di sini", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Tampilkan Hasil Prediksi Besar-Besar
    cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # Tampilkan Bar Probabilitas (Opsional - biar keren)
    y_offset = 100
    for i, cls in enumerate(CLASSES):
        p = probs[0][i].item()
        text = f"{cls}: {p*100:.1f}%"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(frame, (120, y_offset-10), (120 + int(p*200), y_offset+5), (0, 255, 0), -1)
        y_offset += 30

    cv2.imshow("Hand Gesture Recognition - Anxten", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()