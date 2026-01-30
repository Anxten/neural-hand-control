# 🧠 Neural Hand Control (Linux/Wayland Edition)

Control your Linux laptop media and volume using AI-powered hand gestures. Built with PyTorch and OpenCV.

## 🌟 Features

| Gesture | Action | Implementation |
| :--- | :--- | :--- |
| ✊ **ROCK** | Play / Pause Media | Uses `playerctl` to toggle Spotify/YouTube |
| ✋ **PAPER** | Volume UP (+2%) | Uses `pactl` (PulseAudio) for smooth control |
| ✌️ **SCISSORS** | Mute / Unmute | Uses `pactl` to toggle sound instantly |

> **Note:** Optimized for **Linux (Wayland)** environments where external automation tools like `pyautogui` often fail. This project uses native system calls (`subprocess`) for reliability.

## 🛠️ Requirements

### 1. System Dependencies
```bash
# For Ubuntu/Debian:
sudo apt install playerctl pulseaudio-utils

# For Fedora/RHEL:
sudo dnf install playerctl pulseaudio-utils

# For Arch Linux:
sudo pacman -S playerctl pulseaudio
```

### 2. Python Dependencies
```bash
pip install torch torchvision opencv-python numpy
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Anxten/neural-hand-control.git
cd neural-hand-control
```

### 2. Create Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to avoid conflicts.

```bash
# Create venv
python -m venv venv

# Activate venv (Linux/MacOS)
source venv/bin/activate

# Activate venv (Windows)
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Data Directories
```bash
mkdir -p data/raw/{rock,paper,scissors,none}
mkdir models
```

### 5. Collect Training Data
```bash
python collect_data.py
```
- Press `r` for ROCK gestures
- Press `p` for PAPER gestures  
- Press `s` for SCISSORS gestures
- Press `n` for NONE (empty hand)
- Collect ~500 samples per gesture

### 6. Train the Model
```bash
python train.py
```

### 7. Run Hand Control
```bash
python main.py
```

## 🎯 Usage Tips

- **Good Lighting**: Ensure adequate lighting for better detection
- **Hand Position**: Keep your hand within the green box
- **Confidence Threshold**: System only responds when >90% confident
- **Cooldown System**: 
  - PAPER: 1 second cooldown (for volume control)
  - ROCK/SCISSORS: 2 seconds cooldown (prevents double actions)

## 🧠 Model Architecture
- **Input:** 64x64 RGB images from webcam ROI
- **Model:** Custom CNN with 3 Convolutional Layers + Dropout (50%)
- **Classes:** 4 gestures (none, paper, rock, scissors)
- **Training:** Data augmentation with rotation, brightness, and scaling

## 🔧 Troubleshooting

**Camera not working:**
```bash
# Check camera permissions
ls /dev/video*
```

**Audio commands not working:**
```bash
# Test playerctl
playerctl status

# Test pactl
pactl info
```

**Model not found:**
- Make sure you've run `python train.py` first
- Check if `models/hand_gesture_cnn.pth` exists

---
*Created by Allan Bendatu - Informatics Student*