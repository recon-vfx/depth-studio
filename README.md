# 🦇 Depth Studio

> Turn any video into depth maps and normal maps for cinematics and VFX. 

Depth Studio is a high-performance Python application built with **PySide6** and **ONNX Runtime (DirectML / CUDA)**. It runs entirely locally on your DirectX 12 GPU (NVIDIA, AMD, or Intel) without requiring heavy frameworks like PyTorch or Transformers.

---

## ✨ Features

* **Multi-Model Support:** 
  * **Video Depth Anything:** Built for temporally consistent video depth without flickering.
  * **Depth Anything V3 / V2:** Delivers ultra-sharp single-frame details (Small, Base, Large, Giant).
* **Dual-Model Blending:** Combine a stable primary model (like Video Depth Anything) with a sharp detail model to capture fine edges without temporal boiling.
* **Normal Map Generation:** Instantly compute smooth surface normal maps directly from depth.
* **Flexible Export Formats:** Export as 16-bit PNG sequences, 32-bit EXR floating-point sequences, H.264/H.265 MP4s, Apple ProRes (422 HQ / 4444 / XQ), or Lossless FFV1 MKV.
* **GPU Accelerated:** Powered by ONNX Runtime with native DirectML support.

---

## 📥 Download Compiled Release (Windows)

If you prefer to run the pre-compiled standalone application without setting up a Python environment:

* **[Download Depth Studio v1.0.0 (Google Drive / OneDrive / Dropbox Link Here)]**(YOUR_CLOUD_STORAGE_LINK_HERE)

*(Note: Hosted externally on cloud storage to ensure fast, hassle-free downloads).*

---

## 🚀 Running from Source

If you want to run or modify the code directly:

### 1. Prerequisites
* Python 3.10 – 3.12 installed on Windows.

### 2. Install Dependencies
Open your terminal inside the project folder and run:


pip install onnxruntime-directml numpy opencv-python PySide6 imageio-ffmpeg
