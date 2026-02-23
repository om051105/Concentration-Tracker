# Concentration Tracker 🎯

A real-time concentration monitoring system using computer vision to track facial position, eye gaze, and blink detection.

## 📋 Features

- **Real-time Face Detection**: Tracks your face position using OpenCV's Haar Cascades
- **Eye Gaze Tracking**: Monitors where you're looking (left, center, right)
- **Blink Detection**: Counts and detects blinks with high accuracy
- **Concentration Score**: Provides a dynamic score based on position, gaze, and eye status
- **Visual Feedback**: Real-time display of concentration metrics with color-coded indicators

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/om051105/Concentration-Tracker.git
cd Concentration-Tracker
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

Run the concentration tracker:
```bash
python ml.py
```

**Controls:**
- `q` - Quit the application
- `r` - Recalibrate baseline position

## 📊 How It Works

1. **Calibration Phase**: Look straight at the camera for 3 seconds to set baseline
2. **Tracking**: The system continuously monitors:
   - Face position relative to camera center
   - Eye gaze direction
   - Eye status (open/closed)
   - Blink frequency
3. **Scoring**: Concentration score calculated from:
   - Position score (40% weight)
   - Gaze score (40% weight)
   - Eye status (20% weight)

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- NumPy
- Webcam

## 🤖 Automation Features

This repository includes an automated daily motivation quote system:

- **Daily Updates**: A GitHub Actions workflow automatically updates the README with a new motivational quote every day at midnight UTC
- **Quote Database**: Curated collection of 30+ inspiring quotes that rotate daily
- **Manual Update**: You can also manually update the quote by running:
  ```bash
  python update_quotes.py
  ```
- **Manual Trigger**: The GitHub Actions workflow can be manually triggered from the Actions tab

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💪 Daily Motivation

Get inspired with a new motivational quote every day! This section is automatically updated daily at midnight UTC.

> **Quote of the Day:**
> 
> *"Dream it. Believe it. Build it."*
> 
> — Unknown

---

*Last updated: 2026-02-23 | Automatically updated via GitHub Actions*
