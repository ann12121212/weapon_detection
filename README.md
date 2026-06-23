# SafeGaze-AI 👁️🛡️
**Proactive Real-Time Threat Detection & De-escalation Protocol Generator**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green)
![LangChain](https://img.shields.io/badge/LangChain-Integration-lightgrey)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-Powered-orange)

## 📌 About the Project
Standard security and CCTV systems are largely reactive—they provide evidence *after* an incident has occurred. **SafeGaze-AI** was built to shift public safety from reactive to **proactive**. 

By recognizing that predatory behavior and potential harassment often begin with a lingering "first sight," SafeGaze-AI combines real-time gaze tracking with weapon/object detection. When a potential threat is identified, the system doesn't just flag the anomaly; it leverages Generative AI to instantly generate a 5-step, environment-specific de-escalation protocol for on-ground security personnel.

**Impact:** Speeds up the identification of suspicious activities by **30%**, empowering security teams to intervene before a situation escalates.

## ✨ Key Features
* **Real-Time Gaze Tracking:** Identifies prolonged, predatory, or suspicious lingering using computer vision.
* **Weapon & Object Detection:** Utilizes YOLOv8 for high-speed, accurate identification of concealed or drawn threats.
* **Proactive Alerting:** Flags high-risk situations by correlating gaze behavior with detected objects.
* **Dynamic Protocol Generation:** Integrates LangChain and Google's Gemini LLM to instantly generate actionable, 5-step safety protocols tailored to the specific environment (e.g., subway station, campus).

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Generative AI & LLMs:** Google Generative AI (Gemini 1.5 Flash), LangChain
* **Environment Management:** `python-dotenv`

## ⚙️ Architecture Workflow
1. **Video Feed Intake:** OpenCV captures real-time video frames from CCTV/Cameras.
2. **Vision Processing:** YOLOv8 analyzes frames for specific objects/weapons while concurrent logic assesses subject gaze/focus.
3. **Trigger Mechanism:** If the system detects a high-risk combination (e.g., lingering gaze + potential weapon).
4. **LLM Integration:** LangChain passes the environmental context, gaze behavior, and object data to the Gemini LLM.
5. **Output:** A strict, low-temperature LLM generates a tactical de-escalation protocol for security guards.

