# 🌊 Microplastic Risk Intelligence System

## 🚀 Overview
Microplastics (particles < 5mm) are a major environmental threat affecting marine ecosystems and human health.

This project presents a computer vision-based system that detects microplastic particles from images and evaluates their ecological risk based on morphology and size.

---

## 🎯 Problem Statement
Traditional microplastic identification methods are expensive and require laboratory setups.

This system provides a **low-cost, automated alternative** using image processing and intelligent scoring to:
- Identify particle type
- Estimate size
- Assess ecological impact

---

## 🧠 Key Features

### 🔹 Morphology Classification
Classifies microplastics into:
- **Fiber** (thread-like, highest risk)
- **Fragment** (irregular, medium risk)
- **Film** (sheet-like, lower risk)

---

### 🔹 Size Estimation
- Uses contour detection
- Estimates **Feret Diameter (longest dimension)**
- Represents particle size in micrometers (µm)

---

### 🔹 Ecological Risk Index
Computes a **0–100 risk score** based on:
- Particle shape (fiber > fragment > film)
- Size (smaller particles → higher penetration risk)
- Geometric features (elongation, irregularity)

---

### 🔹 Overall Risk Assessment
Instead of isolated results, the system provides:
- **Final ecological threat score**
- Severity classification (Low / Medium / High)
- Composition breakdown of detected particles

---

## 🛠 Tech Stack
- **Python**
- **OpenCV** (image processing & contour detection)
- **Streamlit** (interactive UI)
- **NumPy** (numerical computation)
- **PIL** (image handling)

---

## ⚙️ System Workflow

1. Upload microplastic image  
2. Preprocess image (grayscale + blur)  
3. Detect edges and contours  
4. Classify particle morphology  
5. Estimate size (Feret diameter)  
6. Compute risk score  
7. Generate overall ecological risk  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py