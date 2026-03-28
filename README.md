# 🌊 Microplastic Morphology Classification & Risk Assessment System
[WhatsApp Image 2026-03-28 at 15 03 41](https://github.com/user-attachments/assets/df6fecc9-032e-4482-86d9-e4dddb57700e)


This Streamlit application analyzes microscopic images of microplastics and classifies them into morphological categories using computer vision techniques.

The system identifies microplastic particles as:

- Fiber (thread-like)
- Fragment (irregular, jagged)
- Film (thin, sheet-like)

After classification, the system computes an **Ecological Threat Index (0–100)** based on particle morphology, size (Feret diameter), and geometric features. It also provides environmental impact insights and mitigation strategies.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** (UI)
- **OpenCV** (Image Processing)
- **NumPy** (Computation)
- **PIL** (Image Handling)

---

## ⚙️ Features

### 🔹 Microplastic Classification
Classifies particles into:
- Fiber → highest ecological risk  
- Fragment → moderate risk  
- Film → lower risk  

---
## SAMPLE
[Watch Demo](https://github.com/chinmayeechaple/microplastic-ai-system/blob/main/Screen%20Recording%202026-03-28%20151514.mp4)


 
### 🔹 Size Estimation
- Uses contour detection  
- Computes **Feret Diameter (longest dimension)**  
- Estimates particle size in micrometers (µm)  

---

### 🔹 Ecological Risk Assessment
Computes a **risk score (0–100)** using:
- Morphology weight  
- Particle size  
- Shape elongation  
- Contour irregularity  

---

### 🔹 Overall Risk Output
- Final **Ecological Threat Index**
- Severity classification:
  - 🔴 High  
  - 🟡 Medium  
  - 🟢 Low  

---

## 🧠 Model

This system uses a **feature-based computer vision model inspired by machine learning principles**.

Instead of training a deep learning model, it performs:

- Edge detection (Canny)
- Line detection (Hough Transform for fibers)
- Contour analysis for fragments and films
- Geometric feature extraction:
  - Aspect ratio  
  - Perimeter-to-area ratio  
  - Shape complexity  

These features are combined using a **weighted scoring function** to simulate intelligent decision-making similar to ML models.

---

## 📊 Dataset

This project does not rely on a pre-trained dataset. Instead, it works on:

- Microscopic images of microplastics  
- High-resolution synthetic or real-world sample images  

For testing and demonstration, images were selected based on:
- Clear contrast  
- Distinct particle shapes  
- Minimal noise  

Future versions can integrate datasets such as:
- Microplastic morphology datasets  
- Marine pollution image datasets  
- Research datasets from environmental studies  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/microplastic-ai-system.git
cd microplastic-ai-system
