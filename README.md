# SafeGaze-AI 👁️🛡️

> **AI-powered real-time threat detection system that combines a fine-tuned YOLOv8 model, gaze estimation, and Google's Gemini to assist security personnel with proactive decision making.**

---

## 📌 Overview

Traditional surveillance systems are primarily reactive—they record incidents but cannot actively assist security personnel during an ongoing situation.

**SafeGaze-AI** enhances surveillance by combining **Computer Vision** and **Generative AI** to detect potential threats and instantly generate actionable security protocols.

The system simultaneously performs:

* Real-time weapon detection using a **fine-tuned YOLOv8 model (`best.pt`)**
* Head pose and gaze estimation using **MediaPipe Face Mesh**
* Live monitoring through a Flask dashboard
* AI-generated de-escalation protocols using **Google Gemini + LangChain**
* Real-time Socket.IO alerts

---

# 🚀 Features

## 🔫 Fine-Tuned YOLOv8 Weapon Detection

* Custom trained YOLOv8 model (`best.pt`)
* Detects multiple dangerous objects including:

  * Knife
  * Gun
  * Pistol
  * Rifle
  * Scissors
* Real-time inference
* Confidence-based detection
* Optimized for webcam/CCTV feeds

---

## 👁️ Gaze Estimation

MediaPipe Face Mesh is used to estimate head orientation and gaze direction.

Detected behaviors include:

* Looking Left
* Looking Right
* Looking Up
* Looking Down
* Forward gaze (Gaze Alert)

This provides additional behavioral context alongside weapon detection.

---

## 🤖 AI Security Protocol Generator

Whenever a weapon is detected, the system sends contextual information to **Google Gemini** using **LangChain**.

The LLM generates a structured **5-step de-escalation protocol** based on:

* detected weapon
* gaze behavior
* monitored location

If the Gemini API is unavailable, SafeGaze automatically switches to a predefined fail-safe emergency protocol.

---

## 🌐 Real-Time Web Dashboard

Built using Flask and Flask-SocketIO.

The dashboard provides:

* Live camera stream
* Weapon alerts
* Gaze alerts
* Interactive AI chatbot
* Live detection updates

---

## ⚡ Performance Monitoring

The application continuously displays:

* FPS
* YOLO preprocessing time
* YOLO inference time
* YOLO postprocessing time

to help evaluate real-time deployment performance.

---

# 🏗️ Architecture

```
               Webcam / CCTV
                      │
                      ▼
             OpenCV Video Capture
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 MediaPipe Face Mesh          Fine-Tuned YOLOv8
 (Head Pose Estimation)       (best.pt)
        │                           │
        ▼                           ▼
 Gaze Behaviour              Weapon Detection
        │                           │
        └─────────────┬─────────────┘
                      ▼
          Threat Context Generation
                      │
                      ▼
        LangChain Prompt Engineering
                      │
                      ▼
      Google Gemini (LLM Response)
                      │
                      ▼
      5-Step Security Protocol
                      │
                      ▼
      Flask + Socket.IO Dashboard
```

---

# 🛠️ Tech Stack

| Category                | Technologies        |
| ----------------------- | ------------------- |
| Programming Language    | Python              |
| Object Detection        | Fine-Tuned YOLOv8   |
| Computer Vision         | OpenCV              |
| Face Tracking           | MediaPipe Face Mesh |
| Backend                 | Flask               |
| Real-Time Communication | Flask-SocketIO      |
| LLM Framework           | LangChain           |
| Generative AI           | Google Gemini       |
| Environment Variables   | python-dotenv       |

---

# 📂 Project Structure

```
weapon_detection/
│
├── best.pt                  # Fine-tuned YOLOv8 model
├── run_weapon.py            # Main application
├── home.html
├── chatbot.html
├── home.css
├── final-weapon-det-pro.ipynb
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ann12121212/weapon_detection.git
cd weapon_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GOOGLE_API_KEY=your_google_api_key
```

Run the application

```bash
python run_weapon.py
```

Open your browser

```
http://localhost:5000
```

---

# 🔄 Workflow

1. Capture live webcam frames using OpenCV.
2. Estimate gaze direction with MediaPipe Face Mesh.
3. Detect weapons using the fine-tuned YOLOv8 model.
4. Update threat context.
5. Send context to Gemini through LangChain.
6. Generate a 5-step security protocol.
7. Display alerts and live video on the dashboard.

---

# 💬 AI Security Assistant

The chatbot can generate context-aware responses such as:

```
Threat Detected:
Weapon: Knife
Location: Campus Entrance
Gaze: Looking Left

Generated Protocol:

1. Notify nearby security personnel.
2. Track the suspect using surrounding cameras.
3. Maintain a safe distance and redirect civilians.
4. Prepare containment if the threat escalates.
5. Contact emergency authorities immediately if required.
```

---

# 📊 Performance

The application displays:

* Live FPS
* Preprocessing time
* Inference latency
* Postprocessing latency

allowing real-time performance evaluation during deployment.

---

# 🎯 Applications

* Educational Institutions
* Corporate Offices
* Shopping Malls
* Airports
* Railway Stations
* Smart City Surveillance
* Public Events
* Campus Security

---

# 🔮 Future Improvements

* Multi-camera support
* Object tracking (ByteTrack / DeepSORT)
* Person re-identification
* Threat severity scoring
* SMS/Email alerts
* Incident logging dashboard
* Docker deployment
* Cloud deployment
* Mobile notification support

---

# 📸 Demo

Add screenshots or GIFs of:

* Live weapon detection
* Gaze estimation
* Dashboard interface
* AI-generated protocol
* Real-time alerts

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

# 📄 License

This project is intended for educational and research purposes.

---

# 👩‍💻 Author

**Samriddhi Khare**

Integrated M.Tech (Mathematics & Computing)

Indian Institute of Technology (ISM) Dhanbad

GitHub: https://github.com/ann12121212
