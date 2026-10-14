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

* **[Download Depth Studio v1.0.0 (Google Drive / OneDrive / Dropbox)](YOUR_CLOUD_STORAGE_LINK_HERE)**

*(Note: Hosted externally on cloud storage for fast, hassle-free downloads).*

---

## 🧠 Model Downloads (.onnx files)

Depth Studio requires pre-exported `.onnx` model files to run. Place them inside a folder named `models` right next to `DepthStudio.py`. 

You can download compatible ONNX exports from the official repositories and community hubs:

* **Video Depth Anything (Encoder + Head Pairs):** [GitHub - Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
* **Depth Anything V2 / V3 ONNX Files:** [Hugging Face - Depth-Anything-V3-ONNX (Till Beemelmanns)](https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX) or [GitHub - Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

*(Note: For Video Depth Anything, make sure to keep both the `*_encoder_*` and matching `*_head_*` `.onnx` files together in the same `models` folder).*

---

## 🚀 Running from Source

If you want to run or modify the code directly:

### 1. Prerequisites
* Python 3.10 – 3.12 installed on Windows.

### 2. Install Dependencies
Open your terminal inside the project folder and run:
```bash
pip install onnxruntime-directml numpy opencv-python PySide6 imageio-ffmpeg
(Do not install plain onnxruntime alongside onnxruntime-directml, as they conflict).

3. Add AI Models
Create a folder named models next to DepthStudio.py and drop your .onnx model files inside it.

4. Run the App
- open your terminal in the folder with depth studio
python DepthStudio.py
📜 Licenses
Please review the respective model card licenses for any .onnx models you use (e.g., Depth Anything V2 Large is licensed under CC-BY-NC-4.0 for non-commercial use; V2 Small/Base, V3, and Video Depth Anything are Apache-2.0).