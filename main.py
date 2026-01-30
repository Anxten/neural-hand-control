import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import time
import subprocess

# --- 1. KONFIGURASI ---
MODEL_PATH = "models/hand_gesture_cnn.pth"
CLASSES = ['none', 'paper', 'rock', 'scissors']
BOX_SIZE = 250
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DEFINISI MODEL ---
class HandGestureCNN(nn.Module):
    def __init__(self):
        super(HandGestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 3. LOAD MODEL ---
print("🧠 Memuat model AI...")
model = HandGestureCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# --- 4. SIAPKAN VARIABLE KONTROL ---
last_action_time = 0
cap = cv2.VideoCapture(0)

print("📷 Kamera siap! Wayland Mode ON.")

def run_command(cmd_list):
    try:
        subprocess.Popen(cmd_list)
        return True
    except Exception as e:
        print(f"Gagal eksekusi: {e}")
        return False

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    h, w, c = frame.shape
    x1, y1 = int(w/2) - 150, 100
    x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
    roi = frame[y1:y2, x1:x2]
    
    # Prediksi AI
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_tensor = transform(roi_rgb).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(roi_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        class_name = CLASSES[predicted.item()]
        conf_score = confidence.item() * 100

    # --- LOGIKA KONTROL (TUNED V3) ---
    current_time = time.time()
    
    if class_name == 'paper':
        needed_cooldown = 1.0
    else:
        needed_cooldown = 2.0
    
    if conf_score > 90 and (current_time - last_action_time > needed_cooldown):
        
        if class_name == 'rock':
            print("✊ ROCK -> Toggle Media")
            run_command(["playerctl", "play-pause"])
            last_action_time = current_time 
            
        elif class_name == 'paper':
            print("✋ PAPER -> Volume +2%")
            run_command(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+2%"])
            last_action_time = current_time 

        elif class_name == 'scissors':
            print("✌️ SCISSORS -> Mute/Unmute Sound")
            run_command(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
            last_action_time = current_time

    # --- VISUALISASI ---
    color = (0, 255, 0) if conf_score > 90 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    text = f"{class_name.upper()} {conf_score:.1f}%"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    time_passed = current_time - last_action_time
    if time_passed < needed_cooldown:
        ratio = time_passed / needed_cooldown
        bar_width = int(ratio * 200)
        cv2.rectangle(frame, (20, 80), (20 + bar_width, 90), (0, 0, 255), -1)
        cv2.putText(frame, "WAIT...", (230, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(frame, "READY!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Neural Hand Control (Wayland Safe)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()