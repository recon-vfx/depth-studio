"""
 Depth Studio - Made by Reconvfx          (ONNX Runtime + DirectML edition)
Video -> depth maps + normal maps for cinematics. Runs from .onnx model files on ANY DirectX 12 GPU
(NVIDIA / AMD / Intel). No torch, no transformers.

BLEND MODE: pick a second single-frame model under "Detail model". The main model supplies the overall depth and the
temporal stability (use Video Depth Anything for that); the second supplies the fine edges (Depth Anything V3 / V2 Large).

MODEL LIST: models are ranked and shown best first, e.g.
  *** BEST - Depth Anything V3 - Large (sharpest detail, slower)
  **  GOOD - Depth Anything V3 - Base (best balance of speed and quality)
  *   FAST - ... (fastest, lighter detail)
The best model is loaded by default (a model you picked before is remembered in depth_studio.json).

SUPPORTED MODELS (drop the .onnx files in a "models" folder next to this script / the .exe)
  * Depth Anything V2  (Small / Base / Large)       - the files made by your export_model.py
  * Depth Anything V3  (any single-view ONNX export) - fixed-size or dynamic, fp32 or fp16
  * Video Depth Anything (encoder + head pair)       - temporally consistent video depth.
        Keep BOTH files in the same folder, e.g.  vda_vits_encoder_518x910_fp16.onnx
                                                   vda_vits_head_T32_37x65_fp16.onnx
        Only the *_encoder_* file shows up in the model list; the matching *_head_* file is found automatically.

UPDATES: on startup the app checks the newest published GitHub Release of REPO (below). If its tag is higher than
VERSION, a popup offers to open the download page.

INSTALL (Windows, Python 3.10-3.12)
  pip install onnxruntime-directml numpy opencv-python PySide6 imageio-ffmpeg
  (do NOT also install plain "onnxruntime" - they conflict)
RUN
  python DepthStudio.py

LICENSES: check the model card of every .onnx you use. Depth Anything V2 Large is CC-BY-NC-4.0 (non-commercial);
V2 Small, V3 Small/Base and Video Depth Anything Small are Apache-2.0.
"""
import os, sys, time, glob, json, shutil, subprocess, threading, queue, tempfile, traceback
import urllib.request, webbrowser
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # must be set before importing cv2 (EXR export)
APP_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
import numpy as np
import cv2
import onnxruntime as ort
ort.set_default_logger_severity(3)   # only show real errors (hides the harmless "constant fold Sqrt" warnings)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QComboBox,
                               QCheckBox, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QProgressBar,
                               QGroupBox, QFormLayout, QMessageBox, QScrollArea, QListWidget,
                               QListWidgetItem, QSpinBox, QSizePolicy, QPlainTextEdit)

# ---------------------------------------------------------------- version / updates
VERSION = "1.0.0"                      # Change this to "1.0.0" to test your update notification against release v1.0.1
REPO = "recon215/depth-studio"         # GitHub  user/repo  (the repo must be public)


def _vt(v):
    """'v1.2.3' -> (1, 2, 3) so versions compare correctly."""
    return tuple(int(x) for x in str(v).lstrip("vV").split(".") if x.isdigit())


class Updater(QObject):
    """Asks GitHub for the latest published release in a background thread with built-in debug output."""
    found = Signal(str, str, str)   # tag, release notes, page url

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            req = urllib.request.Request("https://api.github.com/repos/%s/releases/latest" % REPO,
                                         headers={"User-Agent": "DepthStudio"})
            with urllib.request.urlopen(req, timeout=8) as r:
                d = json.load(r)
            tag = d.get("tag_name", "")
            if tag and _vt(tag) > _vt(VERSION):
                print("[DEBUG] New version detected! Emitting signal...")
                self.found.emit(tag, (d.get("body") or "")[:800], d.get("html_url", ""))
            else:
                print("[DEBUG] App is already up to date or tag was empty.")
        except Exception as e:
            print("[DEBUG] Updater check failed:", e)
            pass


# ---------------------------------------------------------------- settings file
CFG_PATH = os.path.join(APP_DIR, "depth_studio.json")


def cfg_load():
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def cfg_save(d):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


CFG = cfg_load()
GPU_ID = int(CFG.get("gpu_id", 0))                 # which graphics card DirectML uses
BLEND_STYLE = CFG.get("blend_style", "clean")      # "clean" = edge-snapping, no halos | "sharp" = max fine detail, can halo
MAX_TOKENS = int(CFG.get("max_tokens", 6000))      # image patches per frame (dynamic-size models); GPU cost grows with its square

# ---------------------------------------------------------------- GPU backend
_AV = ort.get_available_providers()
BACKEND = next((p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider") if p in _AV), None)


def make_providers():
    if BACKEND == "DmlExecutionProvider":
        return [("DmlExecutionProvider", {"device_id": GPU_ID}), "CPUExecutionProvider"]
    return ([BACKEND] if BACKEND else []) + ["CPUExecutionProvider"]


DETAIL = [("Fast (392)", 392), ("Balanced (518)", 518), ("High (644)", 644), ("Ultra (756)", 756)]
FORMATS = ["16-bit PNG sequence", "EXR sequence (32-bit float)", "MP4 - High quality (H.264)",
           "MP4 - 10-bit (H.265, smoothest gradients)", "MOV - ProRes 422 HQ (10-bit, best for editors)",
           "MOV - ProRes 4444 (10-bit, full color, no chroma loss)",
           "MOV - ProRes 4444 XQ (10-bit, highest ProRes quality)", "MKV - Lossless FFV1 (16-bit when grayscale)"]
F_PNG, F_EXR, F_H264, F_H265, F_PRO422, F_PRO4444, F_PROXQ, F_FFV1 = FORMATS
PRORES = {F_PRO422: (3, "yuv422p10le"), F_PRO4444: (4, "yuv444p10le"), F_PROXQ: (5, "yuv444p10le")}  # profile, pixel format
DEEP = (F_H265, F_FFV1) + tuple(PRORES)   # formats that can carry more than 8 bits: we feed them 16-bit data
COLORMAPS = {"Grayscale": None, "Inferno": cv2.COLORMAP_INFERNO, "Turbo": cv2.COLORMAP_TURBO,
             "Magma": cv2.COLORMAP_MAGMA, "Viridis": cv2.COLORMAP_VIRIDIS, "Jet": cv2.COLORMAP_JET}
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".wmv")
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
GPU_ERR = ("887A", "8007000E", "DEVICE_REMOVED", "DEVICE_HUNG", "OUTOFMEMORY", "out of memory", "Not enough memory", "bad_alloc")
BROWSE = "Browse for .onnx file..."
CLEAN_R = [0, 4, 8, 12, 18, 24]
NO_BLEND = "None (single model)"
OUT_TYPES = ["Auto-detect", "Disparity (bright = near)", "Depth (bright = far) - convert"]


# ---------------------------------------------------------------- model naming / discovery
def _low(stem):
    return stem.lower().replace("-", "_")


def is_vda_enc(stem):
    low = _low(stem)
    return "_encoder" in low and ("vda" in low or "video" in low)


def is_vda_head(stem):
    low = _low(stem)
    return "_head" in low and ("vda" in low or "video" in low)


def is_da3(stem):
    low = _low(stem)
    return "da3" in low or "depth_anything_3" in low or "depthanything3" in low or "depth_anything_v3" in low


def guess_depth_output(path):
    """Depth Anything 3 any-view exports output *depth* (far = big). Mono exports and V2 output disparity (near = big).
    This is only a guess from the file name - the 'Model output' box lets you override it."""
    low = _low(os.path.basename(path))
    return is_da3(low) and "mono" not in low


SIZE_RANK = (("giant", 4), ("large", 3), ("vitl", 3), ("base", 2), ("vitb", 2), ("small", 1), ("vits", 1))
SIZE_NAME = {4: "Giant", 3: "Large", 2: "Base", 1: "Small"}
TIER = {4: "★★★ BEST", 3: "★★★ BEST", 2: "★★ GOOD", 1: "★ FAST"}
NOTE = {4: "top quality, heaviest", 3: "sharpest detail, slower", 2: "best balance of speed and quality",
        1: "fastest, lighter detail"}


def size_rank(low, default=1):
    return next((r for k, r in SIZE_RANK if k in low), default)


def sort_key(stem):
    low = _low(stem)
    fam = 3 if is_da3(stem) else 2 if is_vda_enc(stem) else 1 if any(k in low for k, _ in SIZE_RANK) else 0
    return (fam == 0, -size_rank(low) if fam else 0, -fam)


def pretty(stem):
    low = _low(stem)
    r = size_rank(low)
    if is_vda_enc(stem):
        name, extra = "Video Depth Anything - %s" % SIZE_NAME[r], ", temporal, best for video"
    elif is_da3(stem):
        tag = " mono" if "mono" in low else " metric" if "metric" in low else ""
        r = size_rank(low, 2)
        name, extra = "Depth Anything V3 - %s%s" % (SIZE_NAME[r], tag), ""
    elif any(k in low for k, _ in SIZE_RANK):
        name, extra = "Depth Anything V2 - %s" % SIZE_NAME[r], ""
    else:
        return stem
    return "%s - %s (%s%s)" % (TIER[r], name, NOTE[r], extra)


def find_vda_head(enc_path):
    d, base = os.path.split(enc_path)
    low = base.lower()
    prefix = low.split("_encoder")[0]
    prec = "fp16" if "fp16" in low else "fp32" if "fp32" in low else ""
    cands = [p for p in glob.glob(os.path.join(d, "*.onnx")) if os.path.basename(p).lower().startswith(prefix + "_head")]
    cands.sort(key=lambda p: (prec not in os.path.basename(p).lower(), len(p)))
    return cands[0] if cands else None


def unique_label(models, label, stem):
    return label if label not in models else "%s [%s]" % (label, stem)


def find_models():
    dirs = [APP_DIR, os.path.join(APP_DIR, "models")]
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        dirs += [sys._MEIPASS, os.path.join(sys._MEIPASS, "models")]
    paths = []
    for d in dirs:
        paths += glob.glob(os.path.join(d, "*.onnx"))
    out = {}
    stem_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    for f in sorted(set(paths), key=lambda p: (sort_key(stem_of(p)), -os.path.getsize(p))):
        stem = stem_of(f)
        if is_vda_head(stem):
            continue
        out[unique_label(out, pretty(stem), stem)] = f
    return out


def depth_to_disp(d):
    pos = d[d > 0]
    floor = float(np.percentile(pos[::7], 1)) * 0.1 if pos.size else 1e-3
    return (1.0 / np.maximum(d, max(floor, 1e-6))).astype(np.float32)


def fit_box(rgb, W, H):
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb)
    if abs(np.log((w / h) / (W / H))) < 0.06:
        interp = cv2.INTER_AREA if W * H < w * h else cv2.INTER_CUBIC
        return cv2.resize(rgb, (W, H), interpolation=interp), (0, 0, W, H)
    k = min(W / w, H / h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    return cv2.copyMakeBorder(r, y0, H - nh - y0, x0, W - nw - x0, cv2.BORDER_REPLICATE), (x0, y0, nw, nh)


def fit_scale_shift(src, ref):
    x = np.asarray(src)[:, ::4, ::4].astype(np.float64).ravel()
    y = np.asarray(ref)[:, ::4, ::4].astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    if not np.isfinite(a) or a < 0.05 or a > 20:
        a, b = 1.0, float(y.mean() - x.mean())
    return float(a), float(b)


def fuse_detail(d1, d2, amount):
    h, w = d1.shape
    if d2.shape != d1.shape:
        d2 = cv2.resize(np.ascontiguousarray(d2), (w, h), interpolation=cv2.INTER_CUBIC)
    k = max(1, int(round(max(h, w) / 160)))
    sm = lambda x: cv2.resize(np.ascontiguousarray(x), (max(2, w // k), max(2, h // k)),
                              interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    s1, s2 = sm(d1), sm(d2)
    with np.errstate(all="ignore"):
        corr = np.corrcoef(s1, s2)[0, 1]
    if corr < 0:
        d2 = depth_to_disp(d2) if d2.min() > 0 else -d2
        s2 = sm(d2)
    A = np.stack([s2, np.ones_like(s2)], 1)
    (a, b), *_ = np.linalg.lstsq(A, s1, rcond=None)
    if not np.isfinite(a) or a <= 0:
        a, b = 1.0, float(s1.mean() - s2.mean())
    d2 = d2 * np.float32(a) + np.float32(b)
    if BLEND_STYLE == "sharp":
        sigma = max(1.5, w / 320.0)
        lo1 = cv2.GaussianBlur(d1, (0, 0), sigma)
        lo2 = cv2.GaussianBlur(d2, (0, 0), sigma)
        return np.ascontiguousarray(d1 + amount * ((d2 - lo2) - (d1 - lo1)), np.float32)
    lo, hi = np.percentile(d1[::2, ::2], (1, 99))
    sc = max(float(hi - lo), 1e-6)
    g, src = ((d2 - lo) / sc).astype(np.float32), ((d1 - lo) / sc).astype(np.float32)
    snapped = guided_filter(g, src, max(2, int(w / 60)), 1e-4) * sc + lo
    return np.ascontiguousarray(d1 + amount * (snapped - d1), np.float32)


def predict_blend(e1, e2, rgb, size, tta, amount, out_hw=None):
    d = e1.predict(rgb, size, tta, out_hw=out_hw)
    if e2 is not None and amount > 0 and e2.ready():
        d = fuse_detail(d, e2.predict(rgb, size, tta, out_hw=d.shape), amount)
    return d


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.sess = self.enc = self.head = None
        self.path = self.head_path = None
        self.inp = self.out = None
        self.dt = self.head_dt = np.float32
        self.u8 = self.nhwc = self.rank5 = False
        self.fixed = None
        self.T = 32
        self.enc_in, self.enc_outs, self.head_ins, self.head_out = None, [], [], None
        self.out_mode = 0
        self.last_hw, self.provider = (0, 0), "-"

    @property
    def is_video(self):
        return self.head_path is not None

    def unload(self):
        with self.lock:
            self.sess = self.enc = self.head = None
            self.path = self.head_path = None

    def load(self, path):
        with self.lock:
            self.sess = self.enc = self.head = None
            self.path, self.head_path = path, None
            try:
                if is_vda_enc(os.path.splitext(os.path.basename(path))[0]):
                    hp = find_vda_head(path)
                    if not hp:
                        raise RuntimeError("Found the Video Depth Anything encoder but no matching *_head_*.onnx "
                                           "in the same folder. Both files must sit together.")
                    self.head_path = hp
                self._create()
            except Exception:
                self.path = self.head_path = None
                raise

    def _session(self, path):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if BACKEND == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            return ort.InferenceSession(path, so, providers=make_providers())
        except Exception as e:
            if BACKEND is None:
                raise
            so2 = ort.SessionOptions()
            so2.log_severity_level = 3
            return ort.InferenceSession(path, so2, providers=["CPUExecutionProvider"])

    def _read_input(self, i):
        self.u8 = "uint8" in i.type
        self.dt = np.float16 if "float16" in i.type else np.float32
        shp = list(i.shape)
        self.rank5 = len(shp) == 5
        core = shp[-3:]
        self.nhwc = len(core) == 3 and core[-1] == 3 and core[0] != 3
        hw = core[:2] if self.nhwc else core[1:]
        ints = all(isinstance(v, int) and v > 0 for v in hw)
        self.fixed = (int(hw[1]), int(hw[0])) if ints else None

    def _create(self):
        if self.is_video:
            return self._create_video()
        sess = self._session(self.path)
        i = sess.get_inputs()[0]
        self.inp = i.name
        outs = [o.name for o in sess.get_outputs()]
        self.out = (next((n for n in outs if n.lower() == "depth"), None)
                    or next((n for n in outs if "depth" in n.lower() and "conf" not in n.lower()), None) or outs[0])
        self._read_input(i)
        self.provider = sess.get_providers()[0]
        self.sess = sess

    def _create_video(self):
        enc, head = self._session(self.path), self._session(self.head_path)
        i = enc.get_inputs()[0]
        self._read_input(i)
        if not self.fixed:
            raise RuntimeError("The Video Depth Anything encoder must have a fixed input size.")
        hin, eout = head.get_inputs(), enc.get_outputs()
        if len(hin) != len(eout):
            raise RuntimeError("Encoder gives %d feature tensors but the head wants %d - these two files don't belong together."
                               % (len(eout), len(hin)))
        t = hin[0].shape[0]
        self.T = int(t) if isinstance(t, int) and t > 1 else 32
        self.head_dt = np.float16 if "float16" in hin[0].type else np.float32
        self.enc_in, self.enc_outs = i.name, [o.name for o in eout]
        self.head_ins, self.head_out = [x.name for x in hin], head.get_outputs()[0].name
        self.provider = enc.get_providers()[0]
        self.enc, self.head = enc, head

    def _alive(self):
        return (self.enc is not None and self.head is not None) if self.is_video else self.sess is not None

    def _ensure(self):
        if self.path is None:
            raise RuntimeError("No model loaded yet.")
        if not self._alive():
            self._create()

    def _guard(self, fn):
        try:
            return fn()
        except Exception as e:
            if any(k in str(e) for k in GPU_ERR):
                self.sess = self.enc = self.head = None
                raise RuntimeError("The GPU ran out of memory or the driver reset. Lower the Detail level "
                                   "(or lower 'max_tokens' in depth_studio.json) and try again.")
            raise

    def device_label(self):
        return {"DmlExecutionProvider": "GPU - DirectML (adapter %d)" % GPU_ID, "CUDAExecutionProvider": "GPU - CUDA",
                "CoreMLExecutionProvider": "Apple CoreML", "CPUExecutionProvider": "CPU (slow)"}.get(self.provider, self.provider)

    def describe(self):
        bits = [self.device_label()]
        if self.is_video:
            bits.append("temporal, %d-frame window" % self.T)
        if self.fixed:
            bits.append("fixed %dx%d" % self.fixed)
        return " | ".join(bits)

    def ready(self):
        return self.path is not None

    def depth_out(self):
        if self.out_mode == 2:
            return True
        return self.out_mode == 0 and not self.is_video and guess_depth_output(self.path or "")

    def _feed(self, img):
        if self.u8:
            x = img[None]
        else:
            x = (img.astype(np.float32) / 255 - MEAN) / STD
            x = x[None] if self.nhwc else x.transpose(2, 0, 1)[None]
        if self.rank5:
            x = x[:, None]
        return np.ascontiguousarray(x).astype(np.uint8 if self.u8 else self.dt, copy=False)

    def infer_once(self, rgb, size):
        h, w = rgb.shape[:2]
        with self.lock:
            self._ensure()
            box = None
            if self.fixed:
                W, H = self.fixed
                img, box = fit_box(rgb, W, H)
            else:
                s = size / min(h, w)
                W, H = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
                n = (W // 14) * (H // 14)
                if n > MAX_TOKENS:
                    f = (MAX_TOKENS / n) ** .5
                    W, H = max(14, int(W * f / 14) * 14), max(14, int(H * f / 14) * 14)
                img = cv2.resize(np.ascontiguousarray(rgb), (W, H),
                                 interpolation=cv2.INTER_AREA if W < w else cv2.INTER_CUBIC)
            x = self._feed(img)
            o = self._guard(lambda: self.sess.run([self.out], {self.inp: x})[0])
            self.last_hw = (W, H)
        o = self._clean(np.asarray(o, np.float32))
        o = o.reshape(o.shape[-2], o.shape[-1])
        if self.depth_out():
            o = depth_to_disp(o)
        if box:
            x0, y0, bw, bh = box
            o = o[y0:y0 + bh, x0:x0 + bw]
        return o

    @staticmethod
    def _clean(o):
        bad = ~np.isfinite(o)
        if bad.any():
            if bad.mean() > .5:
                raise RuntimeError("The model returned invalid numbers. Lower the Detail level (or use an fp32 model).")
            o = np.nan_to_num(o, nan=0., posinf=0., neginf=0.)
        return o

    def predict(self, rgb, size, tta=False, out_hw=None):
        h, w = out_hw or rgb.shape[:2]
        if self.is_video:
            d = self.vda_single(rgb)
        else:
            d = self.infer_once(rgb, size)
            if tta:
                d2 = self.infer_once(np.ascontiguousarray(rgb[:, ::-1]), size)[:, ::-1]
                lo1, hi1 = np.percentile(d[::2, ::2], (1, 99))
                lo2, hi2 = np.percentile(d2[::2, ::2], (1, 99))
                d2 = (d2 - lo2) / max(hi2 - lo2, 1e-6) * (hi1 - lo1) + lo1
                d = .5 * (d + d2)
        if d.shape != (h, w):
            d = cv2.resize(np.ascontiguousarray(d), (w, h), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(d, np.float32)

    def vda_encode(self, rgb):
        with self.lock:
            self._ensure()
            W, H = self.fixed
            img, box = fit_box(rgb, W, H)
            x = self._feed(img)
            outs = self._guard(lambda: self.enc.run(self.enc_outs, {self.enc_in: x}))
            self.last_hw = (W, H)
        return [np.asarray(o).astype(np.float16) for o in outs], box

    def vda_head(self, feats):
        stacked = {n: np.concatenate([f[k] for f in feats], 0).astype(self.head_dt) for k, n in enumerate(self.head_ins)}
        with self.lock:
            self._ensure()
            o = self._guard(lambda: self.head.run([self.head_out], stacked)[0])
        o = self._clean(np.asarray(o, np.float32))
        o = o.reshape(o.shape[0], o.shape[-2], o.shape[-1])
        if self.depth_out():
            o = np.stack([depth_to_disp(f) for f in o])
        return o

    def vda_single(self, rgb):
        f, (x0, y0, bw, bh) = self.vda_encode(rgb)
        return self.vda_head([f] * self.T)[0][y0:y0 + bh, x0:x0 + bw]


def guided_filter(guide, src, r, eps=1e-3):
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mI, mp = box(guide), box(src)
    a = (box(guide * src) - mI * mp) / (box(guide * guide) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * guide + box(b)


def base_norm(d, lo, hi):
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def depth_to_normals(n, s):
    h, w = n.shape
    d = cv2.GaussianBlur(n, (0, 0), max(0.8, w / 1920.0))
    k = s["nstrength"] * w
    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * k
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * k
    nx, ny = -dx, (-dy if s["nflip"] else dy)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    out = np.dstack((nx * inv, ny * inv, inv)) * 0.5 + 0.5
    return np.ascontiguousarray(out, np.float32)


def finish(n, s, rgb=None):
    k = s["clean"]
    n = np.ascontiguousarray(n, np.float32)
    if k > 0:
        if rgb is not None:
            r = max(1, int(round(CLEAN_R[k] * n.shape[0] / 1080)))
            guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            n = np.clip(guided_filter(guide, n, r), 0, 1).astype(np.float32)
        else:
            n = cv2.bilateralFilter(n, 2 * k + 3, 0.1, 2 + 3 * k)
    n = np.clip((n - s["black"]) / max(s["white"] - s["black"], 1e-3), 0, 1)
    n = n ** s["gamma"]
    if s["mode"] == 1:
        return depth_to_normals(n.astype(np.float32), s)
    return (1 - n if s["invert"] else n).astype(np.float32)


class Temporal:
    def __init__(self, a):
        self.a = a
        self.prev = self.pg = self.lo = self.hi = self.small = None

    def step(self, d, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
        if self.small is not None and float(np.abs(small - self.small).mean()) > .16:
            self.prev = self.pg = self.lo = self.hi = None
        self.small = small
        lo, hi = np.percentile(d[::4, ::4], (1, 99))
        a = self.a
        self.lo = lo if self.lo is None else a * self.lo + (1 - a) * lo
        self.hi = hi if self.hi is None else a * self.hi + (1 - a) * hi
        n = base_norm(d, self.lo, self.hi)
        if self.prev is not None and self.prev.shape == n.shape:
            diff = cv2.GaussianBlur(np.abs(gray - self.pg), (0, 0), 3)
            wgt = (a * .85 * np.exp(-diff / .05)).astype(np.float32)
            n = (wgt * self.prev + (1 - wgt) * n).astype(np.float32)
        self.prev, self.pg = n, gray
        return n


def colorize(n, cmap):
    if n.ndim == 3:
        return (np.clip(n, 0, 1) * 255 + .5).astype(np.uint8)
    u8 = (np.clip(n, 0, 1) * 255 + .5).astype(np.uint8)
    if COLORMAPS[cmap] is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(cv2.applyColorMap(u8, COLORMAPS[cmap]), cv2.COLOR_BGR2RGB)


def gray16(n):
    return (np.clip(n, 0, 1) * 65535 + .5).astype(np.uint16)


def to_bgr(a):
    return np.ascontiguousarray(a[..., ::-1]) if a.ndim == 3 else a


def thumb(rgb, long_side=960):
    h, w = rgb.shape[:2]
    k = min(1., long_side / max(h, w))
    return rgb if k >= 1 else cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)


def save_image(path, arr):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", arr)
    if not ok:
        raise RuntimeError("Couldn't write %s%s" % (os.path.basename(path),
                           "  (this OpenCV build can't write EXR - use the PNG sequence)" if path.lower().endswith(".exr") else ""))
    buf.tofile(path)


def to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def fmt_time(sec):
    sec = max(0, int(sec))
    return "%02d:%02d" % (sec // 60, sec % 60)


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class FFWriter:
    def __init__(self, path, w, h, fps, fmt, quality, mode="rgb24", src=None, ss=0.0, dur=0.0):
        exe = find_ffmpeg()
        if not exe:
            raise RuntimeError("ffmpeg not found. Run:  pip install imageio-ffmpeg")
        self.path, self.mode = path, mode
        self.log = tempfile.TemporaryFile()
        pix_in = {"gray16": "gray16le", "rgb48": "rgb48le"}.get(mode, "rgb24")
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_in,
               "-s", "%dx%d" % (w, h), "-r", "%.5f" % fps, "-i", "-"]
        if src:
            cmd += ["-ss", "%.3f" % ss, "-t", "%.3f" % dur, "-i", src]
        cmd += ["-map", "0:v"] + (["-map", "1:a:0?"] if src else [])
        vf = ["-vf", "scale=out_color_matrix=bt709,pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        if fmt == F_H264:
            cmd += vf + tags + ["-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-crf", str([17, 12, 8][quality]),
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        elif fmt == F_H265:
            cmd += vf + tags + ["-c:v", "libx265", "-preset", "slow", "-crf", str([18, 12, 6][quality]),
                                "-x265-params", "aq-mode=3", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-movflags", "+faststart"]
        elif fmt in PRORES:
            profile, pix = PRORES[fmt]
            cmd += vf + tags + ["-c:v", "prores_ks", "-profile:v", str(profile), "-vendor", "apl0", "-pix_fmt", pix]
            if quality == 1:
                cmd += ["-bits_per_mb", "4096"]
            elif quality == 2:
                cmd += ["-bits_per_mb", "8192", "-mbs_per_slice", "1"]
        else:
            cmd += ["-c:v", "ffv1", "-level", "3", "-g", "1",
                    "-pix_fmt", {"gray16": "gray16le", "rgb48": "rgb48le"}.get(mode, "bgr0")]
        if src:
            cmd += ["-c:a", "pcm_s16le"] if (fmt in PRORES or fmt == F_FFV1) else ["-c:a", "aac", "-b:a", "256k"]
        cmd.append(path)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.log, creationflags=flags)

    def _err(self):
        try:
            self.p.wait(timeout=5)
        except Exception:
            pass
        self.log.seek(0)
        return self.log.read().decode(errors="ignore")[-800:]

    def write_frame(self, n, cmap):
        if self.mode == "gray16":
            data = gray16(n).astype("<u2").tobytes()
        elif self.mode == "rgb48":
            data = (np.clip(n, 0, 1) * 65535 + .5).astype("<u2").tobytes()
        else:
            data = np.ascontiguousarray(colorize(n, cmap)).tobytes()
        try:
            self.p.stdin.write(data)
        except (BrokenPipeError, OSError):
            raise RuntimeError("ffmpeg stopped: " + self._err())

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait()
        if self.p.returncode != 0:
            raise RuntimeError("ffmpeg error: " + self._err())

    def abort(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.kill(); self.p.wait(timeout=5)
        except Exception:
            pass
        try:
            os.remove(self.path)
        except OSError:
            pass


class LoadWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, engine, path):
        super().__init__()
        self.engine, self.path = engine, path

    def run(self):
        try:
            self.engine.load(self.path)
            self.done.emit(self.path)
        except Exception as e:
            self.failed.emit("Model load failed: %s" % e)


class PreviewWorker(QThread):
    ready = Signal(int, object, object, object, str)
    failed = Signal(str)

    def __init__(self, engine, engine2):
        super().__init__()
        self.engine, self.engine2 = engine, engine2
        self.cv, self.req, self.quit, self.paused = threading.Condition(), None, False, False

    def request(self, req):
        with self.cv:
            self.req = req
            self.cv.notify()

    def stop(self):
        with self.cv:
            self.quit = True
            self.cv.notify()

    def run(self):
        cap, cpath, last, frame, cache, cache2 = None, None, -9, None, None, None
        while True:
            with self.cv:
                while self.req is None and not self.quit:
                    self.cv.wait()
                if self.quit:
                    break
                req, self.req = self.req, None
            if self.paused:
                continue
            try:
                video, idx, size, tta, s = req["video"], req["idx"], req["size"], req["tta"], req["s"]
                if video != cpath:
                    if cap:
                        cap.release()
                    cap, cpath, last, frame, cache, cache2 = cv2.VideoCapture(video), video, -9, None, None, None
                if idx != last or frame is None:
                    if idx != last + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, f = cap.read()
                    if not ok:
                        last, frame = -9, None
                        continue
                    frame, last = cv2.cvtColor(f, cv2.COLOR_BGR2RGB), idx
                h, w = frame.shape[:2]
                k = min(1., 1280 / max(h, w))
                pw, ph = max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)
                src = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA) if k < 1 else frame
                if not self.engine.ready():
                    self.ready.emit(idx, src, None, None, "Waiting for the model...")
                    continue
                key = (video, idx, size, tta, self.engine.path, self.engine.out_mode)
                if cache is None or cache[0] != key:
                    t = time.perf_counter()
                    raw = self.engine.predict(frame, size, tta, out_hw=(ph, pw))
                    cache = (key, raw)
                    info = "Preview: model input %dx%d | %d ms | %s" % (self.engine.last_hw[0], self.engine.last_hw[1],
                                                                       (time.perf_counter() - t) * 1000, self.engine.describe())
                    if self.engine.is_video:
                        info += "  (single-frame preview - the export adds the temporal smoothing)"
                else:
                    info = "Preview (look only, cached) | %s" % self.engine.describe()
                raw = cache[1]
                e2, amt = self.engine2, s.get("blend", 0.)
                if amt > 0 and e2.ready():
                    key2 = (video, idx, size, tta, e2.path, e2.out_mode)
                    if cache2 is None or cache2[0] != key2:
                        t = time.perf_counter()
                        cache2 = (key2, e2.predict(frame, size, tta, out_hw=raw.shape))
                        info += " | detail model %d ms" % ((time.perf_counter() - t) * 1000)
                    raw = fuse_detail(raw, cache2[1], amt)
                lo, hi = np.percentile(raw[::4, ::4], (1, 99))
                n = finish(base_norm(raw, lo, hi), s, src)
                self.ready.emit(idx, src, colorize(n, s["cmap"]), n, info)
            except Exception as e:
                self.failed.emit(str(e))
        if cap:
            cap.release()


class StillWorker(QThread):
    done = Signal(str)

    def __init__(self, engine, video, idx, s, path, engine2=None):
        super().__init__()
        self.engine, self.engine2, self.video, self.idx, self.s, self.path = engine, engine2, video, idx, s, path

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
            ok, f = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("Couldn't read that frame.")
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            s = self.s
            d = predict_blend(self.engine, self.engine2, rgb, s["size"], s["tta"], s.get("blend", 0.))
            lo, hi = np.percentile(d[::4, ::4], (1, 99))
            n = finish(base_norm(d, lo, hi), s, rgb)
            if self.path.lower().endswith(".exr"):
                save_image(self.path, to_bgr(n.astype(np.float32)))
            elif n.ndim == 3 or COLORMAPS[s["cmap"]] is None:
                save_image(self.path, to_bgr(gray16(n)))
            else:
                save_image(self.path, cv2.cvtColor(colorize(n, s["cmap"]), cv2.COLOR_RGB2BGR))
            self.done.emit("Saved " + os.path.basename(self.path))
        except Exception as e:
            self.done.emit("Save failed: %s" % e)


class ExportWorker(QThread):
    progress = Signal(int, str)
    preview = Signal(object, object)
    finished_ok = Signal(str, bool)
    failed = Signal(str)

    def __init__(self, s, jobs, engine, engine2=None):
        super().__init__()
        self.s, self.jobs, self.engine, self.engine2, self.cancel = s, jobs, engine, engine2, False
        self.done, self.total, self.t0 = 0, 1, 0.0

    def run(self):
        try:
            self._run()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run(self):
        jobs = []
        for j in self.jobs:
            cap = cv2.VideoCapture(j["video"])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            start = max(0, j["start"])
            end = n - 1 if j["end"] < 0 else min(j["end"], n - 1)
            if end >= start:
                jobs.append({**j, "start": start, "end": end})
        self.total = max(1, sum(j["end"] - j["start"] + 1 for j in jobs))
        self.t0 = time.time()
        os.makedirs(self.s["out_dir"], exist_ok=True)
        for k, j in enumerate(jobs):
            if self.cancel:
                break
            self._job(j, "Clip %d/%d" % (k + 1, len(jobs)))
        self.finished_ok.emit(self.s["out_dir"], self.cancel)

    def _stream(self, cap, count, w, h):
        e2, amt = self.engine2, self.s.get("blend", 0.)
        for i, rgb, raw in self._stream_base(cap, count, w, h):
            if amt > 0 and e2 is not None and e2.ready():
                raw = fuse_detail(raw, e2.predict(rgb, self.s["size"], self.s["tta"], out_hw=(h, w)), amt)
            yield i, rgb, raw

    def _stream_base(self, cap, count, w, h):
        eng, s = self.engine, self.s
        if eng.is_video:
            yield from self._stream_video(cap, count, w, h)
            return
        for i in range(count):
            if self.cancel:
                return
            ok, f = cap.read()
            if not ok:
                return
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            yield i, rgb, eng.predict(rgb, s["size"], s["tta"], out_hw=(h, w))

    def _stream_video(self, cap, count, w, h):
        eng = self.engine
        T = eng.T
        O = max(2, min(10, T // 3))
        S = T - O
        total, nread, eof, box = count, 0, False, None
        feats, rgbs = {}, {}

        def ensure(upto):
            nonlocal total, nread, eof, box
            while nread <= upto and nread < total and not eof:
                if self.cancel:
                    return
                ok, f = cap.read()
                if not ok:
                    total, eof = nread, True
                    return
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                feats[nread], box = eng.vda_encode(rgb)
                rgbs[nread] = rgb
                nread += 1

        held, start = None, 0
        while start < total and not self.cancel:
            ensure(min(start + T - 1, total - 1))
            if self.cancel or start >= total:
                break
            idx = [min(start + j, total - 1) for j in range(T)]
            d = eng.vda_head([feats[i] for i in idx])
            x0, y0, bw, bh = box
            d = d[:, y0:y0 + bh, x0:x0 + bw]
            is_last = start + T >= total
            if held is not None:
                a, b = fit_scale_shift(d[:O], held)
                d = d * a + b
                ramp = np.linspace(0, 1, O + 2)[1:-1]
                for j in range(O):
                    d[j] = (1 - ramp[j]) * held[j] + ramp[j] * d[j]
            for j in range(T if is_last else S):
                gi = start + j
                if gi >= total:
                    break
                yield gi, rgbs.pop(gi), np.ascontiguousarray(cv2.resize(d[j], (w, h), interpolation=cv2.INTER_CUBIC), np.float32)
            if is_last:
                break
            held = [d[j].copy() for j in range(S, T)]
            for gi in [g for g in feats if g < start + S]:
                del feats[gi]
            start += S

    def _job(self, j, label):
        s = self.s
        cap = cv2.VideoCapture(j["video"])
        if not cap.isOpened():
            raise RuntimeError("Couldn't open " + j["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w, h = int(cap.get(3)), int(cap.get(4))
        normals = s["mode"] == 1
        name = os.path.splitext(os.path.basename(j["video"]))[0] + ("_normal" if normals else "_depth")
        fmt, count = s["format"], j["end"] - j["start"] + 1
        gray = COLORMAPS[s["cmap"]] is None and not normals
        writer = seq_dir = None
        ok_done = False
        try:
            if fmt in FORMATS[2:]:
                ext = ".mov" if fmt in PRORES else ".mkv" if fmt == F_FFV1 else ".mp4"
                deep = fmt in DEEP
                mode = "gray16" if (deep and gray) else "rgb48" if (deep and normals) else "rgb24"
                writer = FFWriter(os.path.join(s["out_dir"], name + ext), w, h, fps, fmt, s["quality"], mode,
                                  src=j["video"] if s["audio"] else None, ss=j["start"] / fps, dur=count / fps)
            else:
                seq_dir = os.path.join(s["out_dir"], name)
                os.makedirs(seq_dir, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, j["start"])
            jobq, err = queue.Queue(maxsize=3), []

            def consumer():
                tmp, lp = Temporal(s["smooth"]), 0.
                try:
                    while True:
                        item = jobq.get()
                        if item is None:
                            break
                        i, rgb, raw = item
                        n = finish(tmp.step(raw, rgb), s, rgb)
                        idx = j["start"] + i
                        if writer is not None:
                            writer.write_frame(n, s["cmap"])
                        elif fmt == F_PNG:
                            save_image(os.path.join(seq_dir, "%s_%06d.png" % (name, idx)), to_bgr(gray16(n)))
                        else:
                            save_image(os.path.join(seq_dir, "%s_%06d.exr" % (name, idx)), to_bgr(n.astype(np.float32)))
                        self.done += 1
                        el = time.time() - self.t0
                        eta = el / self.done * (self.total - self.done)
                        self.progress.emit(int(100 * self.done / self.total), "%s - %d/%d frames - %.2f fps - %s left" % (
                            label, self.done, self.total, self.done / max(el, 1e-6), fmt_time(eta)))
                        if time.time() - lp > .4:
                            lp = time.time()
                            self.preview.emit(thumb(rgb), thumb(colorize(n, s["cmap"])))
                except Exception as ex:
                    err.append(ex)
                    self.cancel = True

            ct = threading.Thread(target=consumer, daemon=True)
            ct.start()

            def put(item):
                while not err:
                    try:
                        jobq.put(item, timeout=.3)
                        return True
                    except queue.Full:
                        pass
                return False

            failed = True
            try:
                for i, rgb, raw in self._stream(cap, count, w, h):
                    if self.cancel:
                        break
                    if not put((i, rgb, raw)):
                        break
                failed = False
            finally:
                if failed or self.cancel or err:
                    try:
                        while True:
                            jobq.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        jobq.put_nowait(None)
                    except queue.Full:
                        pass
                else:
                    put(None)
                ct.join()
            if err:
                raise err[0]
            ok_done = True
        finally:
            cap.release()
            if writer:
                if ok_done and not self.cancel:
                    writer.close()
                else:
                    writer.abort()


class LogBus(QObject):
    line = Signal(str)


LOG = LogBus()


def log(msg):
    LOG.line.emit("[%s] %s" % (time.strftime("%H:%M:%S"), str(msg).rstrip()))


class Tee:
    def __init__(self, orig):
        self.orig = orig

    def write(self, t):
        try:
            if self.orig:
                self.orig.write(t)
        except Exception:
            pass
        if t and t.strip():
            log(t)
        return len(t or "")

    def flush(self):
        try:
            if self.orig:
                self.orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


class StatusLabel(QLabel):
    def __init__(self, *a):
        super().__init__(*a)
        self._last, self._tp = "", 0.

    def setText(self, t):
        super().setText(t)
        if t and t != self._last:
            prog = " frames - " in t
            if not prog or time.time() - self._tp > 5:
                if prog:
                    self._tp = time.time()
                log(t)
        self._last = t


def mk_slider(lo, hi, val, fmt, tip=""):
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(lo, hi)
    sl.setValue(val)
    sl.setToolTip(tip)
    lab = QLabel(fmt(val))
    lab.setMinimumWidth(44)
    sl.valueChanged.connect(lambda v: lab.setText(fmt(v)))
    box = QWidget()
    box.setObjectName("plain")
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(sl)
    lay.addWidget(lab)
    return sl, box


def mk_combo(items, tip=""):
    c = QComboBox()
    c.setToolTip(tip)
    for it in items:
        c.addItem(*it) if isinstance(it, tuple) else c.addItem(it)
    return c


def wide_popup(cb, width=560):
    cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(20)
    cb.view().setMinimumWidth(width)


STYLE = """
QWidget{background:#0b0709;color:#ffe9d6;font-size:13px}
QWidget#plain,QLabel,QCheckBox{background:transparent}
QScrollArea{border:none}
QLabel#title{font-size:24px;font-weight:800;color:#ff7a00}
QLabel#sub{color:#b8754a}
QLabel#status{color:#ffb066}
QLabel#perf{color:#b8754a}
QGroupBox{border:1px solid #4a2410;border-radius:10px;margin-top:14px;padding:14px 8px 8px 8px;background:#120c0f}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px;color:#ff9a2e;font-weight:700}
QPushButton{background:#ff7a00;color:#120804;border:none;border-radius:8px;padding:8px 12px;font-weight:700}
QPushButton:hover{background:#ffa033}
QPushButton:disabled{background:#3a2615;color:#7a5a40}
QPushButton#ghost{background:#1c1210;color:#ffb066;border:1px solid #5a2e12}
QPushButton#ghost:hover{background:#2a1a12}
QComboBox,QLineEdit,QSpinBox,QListWidget{background:#1a1114;border:1px solid #4a2410;border-radius:6px;padding:4px;color:#ffe9d6}
QComboBox QAbstractItemView{background:#1a1114;selection-background-color:#ff7a00;selection-color:#120804}
QListWidget::item:selected{background:#ff7a00;color:#120804}
QLabel#view{background:#050304;border:2px solid #ff7a00;border-radius:10px}
QSlider::groove:horizontal{height:6px;background:#2a1a12;border-radius:3px}
QSlider::sub-page:horizontal{background:#ff7a00;border-radius:3px}
QSlider::handle:horizontal{background:#ffb347;border:2px solid #ff7a00;width:14px;height:14px;margin:-6px 0;border-radius:9px}
QProgressBar{border:1px solid #4a2410;border-radius:6px;text-align:center;background:#1a1114;color:#ffe9d6}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ff5a00,stop:1 #ffb347);border-radius:5px}
QCheckBox::indicator{width:16px;height:16px;border:1px solid #ff7a00;border-radius:4px;background:#1a1114}
QCheckBox::indicator:checked{background:#ff7a00}
QScrollBar:vertical{background:#120c0f;width:10px}
QScrollBar::handle:vertical{background:#5a2e12;border-radius:5px}
QPlainTextEdit#console{background:#050304;color:#ffb066;border:1px solid #4a2410;border-radius:6px;font-family:Consolas,"Cascadia Mono",monospace;font-size:12px}
QToolTip{background:#1a1114;color:#ffe9d6;border:1px solid #ff7a00}
"""


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine, self.engine2 = Engine(), Engine()
        self.models = find_models()
        self._imgs = {}
        self.video, self.nframes, self.fps = None, 0, 24.0
        self.playing = self.exporting = False
        self.worker = self.loader = self.loader2 = self.still = None
        self._pending_load = self._pending2 = None
        self.cur_n, self.cur_idx, self.cur_model = None, 0, None
        self.last_out = CFG.get("out_dir")
        self.setWindowTitle(" Depth Studio v%s - Made by Reconvfx" % VERSION)
        self.resize(1500, 900)
        self.setWindowIcon(QIcon(os.path.join(APP_DIR, "icon.ico")))
        self.setAcceptDrops(True)

        self.pw = PreviewWorker(self.engine, self.engine2)
        self.pw.ready.connect(self.on_ready)
        self.pw.failed.connect(lambda m: self.status.setText("Preview error: " + m))
        self.pw.start()

        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)
        scroll.setMaximumWidth(470)
        panel = QWidget()
        side = QVBoxLayout(panel)
        scroll.setWidget(panel)
        lay.addWidget(scroll, 0)

        t = QLabel(" DEPTH STUDIO - Made by Reconvfx ")
        t.setObjectName("title")
        st = QLabel("turn video into depth + normal maps  🦇  v%s" % VERSION)
        st.setObjectName("sub")
        side.addWidget(t)
        side.addWidget(st)

        g = QGroupBox("🎥 Video")
        v = QVBoxLayout(g)
        b = QPushButton("Open Video...  (or drag & drop)")
        b.clicked.connect(self.open_video)
        self.info = QLabel("No video loaded")
        self.info.setWordWrap(True)
        v.addWidget(b)
        v.addWidget(self.info)
        side.addWidget(g)

        g = QGroupBox("🧠 AI Model")
        f = QFormLayout(g)
        self.model_cb = QComboBox()
        self.model_cb.addItems(list(self.models) + [BROWSE])
        wide_popup(self.model_cb)
        self.model_cb.setToolTip("Models are listed best first. Video Depth Anything = smoothest over time. "
                                 "Depth Anything V3 / V2 Large = sharpest single frames. "
                                 "Models are the .onnx files in the models folder.")
        self.model_cb.activated.connect(self.model_chosen)
        self.outtype = mk_combo(OUT_TYPES, "What the model's raw output means. Auto-detect guesses from the file name "
                                           "(Depth Anything V3 exports usually output depth). If the result looks inverted "
                                           "(near things dark), change this.")
        self.outtype.activated.connect(self.outtype_chosen)
        self.detail = mk_combo(DETAIL, "How much detail the AI sees. Higher = sharper but slower and heavier on the GPU. "
                                       "Ignored by models that only accept one fixed size.")
        self.detail.setCurrentIndex(1)
        self.prev_full = QCheckBox("Preview at full quality (slower)")
        self.prev_full.setToolTip("Off: the preview is capped at Balanced so it stays fast. The export always uses your Detail level.")
        self.dev_lbl = QLabel("Device: -")
        self.dev_lbl.setObjectName("sub")
        self.dev_lbl.setWordWrap(True)
        self.model2_cb = QComboBox()
        wide_popup(self.model2_cb)
        self.model2_cb.setToolTip("Optional second model that adds fine edges. Best pairing: Video Depth Anything as the main model "
                                  "(stable) + Depth Anything V3 / V2 Large here (sharp). Needs more GPU memory and is slower.")
        self.model2_cb.activated.connect(self.model2_chosen)
        self.blend, wbl = mk_slider(0, 100, 70, lambda x: "%d%%" % x, "How much of the detail model's fine structure is mixed in. "
                                                                     "Overall depth and stability always come from the main model.")
        self.refresh_model2()
        f.addRow("AI model", self.model_cb)
        f.addRow("Detail model", self.model2_cb)
        f.addRow("Detail blend", wbl)
        if BACKEND == "DmlExecutionProvider":
            self.gpu_cb = mk_combo(["GPU adapter %d" % i for i in range(4)],
                                   "Laptops often have 2 graphics chips. If it is slow, try another adapter and compare the ms in the preview.")
            self.gpu_cb.setCurrentIndex(min(GPU_ID, 3))
            self.gpu_cb.activated.connect(self.gpu_chosen)
            f.addRow("Graphics card", self.gpu_cb)
        f.addRow("Model output", self.outtype)
        f.addRow("Detail level", self.detail)
        f.addRow(self.prev_full)
        f.addRow(self.dev_lbl)
        side.addWidget(g)

        g = QGroupBox("🎨 Look")
        f = QFormLayout(g)
        self.mode = mk_combo(["Depth map", "Normal map (from depth)"], "Normal map is computed from the cleaned depth.")
        self.cmap = mk_combo(list(COLORMAPS), "Grayscale is best for compositing. (Ignored for normal maps.)")
        self.gamma, w1 = mk_slider(30, 250, 100, lambda x: "%.2f" % (x / 100), "Squeeze or stretch the midtones.")
        self.smooth, w2 = mk_slider(0, 95, 70, lambda x: "%d%%" % x, "Higher = less flicker between frames (export only). "
                                                                     "Video Depth Anything is already stable - 30-50% is plenty.")
        self.black, w3 = mk_slider(0, 90, 0, lambda x: "%d%%" % x, "Push the far background to pure black.")
        self.white, w4 = mk_slider(10, 100, 100, lambda x: "%d%%" % x, "Push the near foreground to pure white.")
        self.clean, w5 = mk_slider(0, 5, 0, lambda x: str(x), "Cleans noise and snaps depth edges to the picture's edges. Higher = stronger.")
        self.nstr, w6 = mk_slider(5, 200, 50, lambda x: "%.2f" % (x / 100), "How steep the surface looks. Higher = stronger relief.")
        self.invert = QCheckBox("Flip (white = far instead of near)")
        self.nflip = QCheckBox("DirectX normals (flip green)")
        self.nflip.setToolTip("Off = OpenGL (Blender, Unity). On = DirectX (Unreal, Maya).")
        self.tta = QCheckBox("Noise reduction (slower, cleaner)")
        self.tta.setToolTip("Runs the AI twice (normal + mirrored) and averages. About 2x slower, noticeably cleaner. "
                            "Not used by Video Depth Anything.")
        f.addRow("Output", self.mode)
        f.addRow("Colormap", self.cmap)
        f.addRow("Contrast", w1)
        f.addRow("Anti-flicker", w2)
        f.addRow("Cut background", w3)
        f.addRow("Cut foreground", w4)
        f.addRow("Edge cleanup", w5)
        f.addRow("Normal strength", w6)
        f.addRow(self.invert)
        f.addRow(self.nflip)
        f.addRow(self.tta)
        side.addWidget(g)

        g = QGroupBox("✂️ Frame range")
        gl = QGridLayout(g)
        self.start_spin, self.end_spin = QSpinBox(), QSpinBox()
        for sp in (self.start_spin, self.end_spin):
            sp.setRange(0, 0)
        b1, b2 = QPushButton("Start = this frame"), QPushButton("End = this frame")
        for bb in (b1, b2):
            bb.setObjectName("ghost")
        b1.clicked.connect(lambda: self.start_spin.setValue(self.slider.value()))
        b2.clicked.connect(lambda: self.end_spin.setValue(self.slider.value()))
        gl.addWidget(QLabel("Start"), 0, 0)
        gl.addWidget(self.start_spin, 0, 1)
        gl.addWidget(b1, 0, 2)
        gl.addWidget(QLabel("End"), 1, 0)
        gl.addWidget(self.end_spin, 1, 1)
        gl.addWidget(b2, 1, 2)
        side.addWidget(g)

        g = QGroupBox("💾 Export")
        v = QVBoxLayout(g)
        self.fmt = mk_combo(FORMATS, "PNG 16-bit or EXR are best for After Effects / Nuke / Blender. "
                                     "Use them for normal maps too - MP4 compression damages normal data. "
                                     "ProRes 4444 / 4444 XQ keep full colour detail (best for colormaps and normal maps).")
        v.addWidget(self.fmt)
        self.vid_opts = QWidget()
        self.vid_opts.setObjectName("plain")
        vo = QFormLayout(self.vid_opts)
        vo.setContentsMargins(0, 0, 0, 0)
        self.quality = mk_combo(["Good (smaller files)", "High", "Max (largest files)"],
                                "Higher = fewer compression artifacts, bigger files. H.264 / H.265: lower CRF. "
                                "ProRes: higher bits per macroblock (Max = the highest the encoder allows, ~380 Mbps at 1080p24). "
                                "Ignored by lossless FFV1.")
        self.quality.setCurrentIndex(1)
        self.audio = QCheckBox("Include original audio")
        vo.addRow("Video quality", self.quality)
        vo.addRow(self.audio)
        v.addWidget(self.vid_opts)
        self.vid_opts.setVisible(False)
        self.fmt.currentIndexChanged.connect(lambda *_: self.vid_opts.setVisible(self.fmt.currentIndex() >= 2))
        v.addWidget(QLabel("Batch queue (empty = export the current video):"))
        self.queue = QListWidget()
        self.queue.setMaximumHeight(90)
        v.addWidget(self.queue)
        qb = QGridLayout()
        for i, (txt, fn) in enumerate([("+ Add videos", self.queue_add_dialog), ("+ Add current", self.queue_add_current),
                                       ("Remove selected", self.queue_remove), ("Clear", self.queue.clear)]):
            bt = QPushButton(txt)
            bt.setObjectName("ghost")
            bt.clicked.connect(fn)
            qb.addWidget(bt, i // 2, i % 2)
        v.addLayout(qb)
        side.addWidget(g)

        self.export_btn = QPushButton("📤 Export")
        self.export_btn.clicked.connect(self.export)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("ghost")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_export)
        self.open_btn = QPushButton("📂 Open output folder")
        self.open_btn.setObjectName("ghost")
        self.open_btn.setEnabled(bool(self.last_out and os.path.isdir(self.last_out)))
        self.open_btn.clicked.connect(self.open_folder)
        self.bar = QProgressBar()
        self.status = StatusLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        for wdg in (self.export_btn, self.cancel_btn, self.open_btn, self.bar, self.status):
            side.addWidget(wdg)
        side.addStretch()

        right = QVBoxLayout()
        lay.addLayout(right, 1)
        pv = QHBoxLayout()
        right.addLayout(pv, 1)
        self.src_lbl, self.dep_lbl = QLabel("🎥 Source"), QLabel("📺 Depth preview")
        for l in (self.src_lbl, self.dep_lbl):
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setObjectName("view")
            l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            l.setMinimumSize(300, 220)
            pv.addWidget(l, 1)
        self.perf = QLabel("")
        self.perf.setObjectName("perf")
        self.perf.setWordWrap(True)
        right.addWidget(self.perf)
        lg = QHBoxLayout()
        lt = QLabel("🖥 Console / log")
        lt.setObjectName("sub")
        cp, cl = QPushButton("Copy"), QPushButton("Clear")
        for bt in (cp, cl):
            bt.setObjectName("ghost")
            bt.setFixedWidth(70)
        lg.addWidget(lt)
        lg.addStretch()
        lg.addWidget(cp)
        lg.addWidget(cl)
        right.addLayout(lg)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("console")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setMinimumHeight(110)
        self.log_view.setMaximumHeight(230)
        LOG.line.connect(self.log_view.appendPlainText)
        cl.clicked.connect(self.log_view.clear)
        cp.clicked.connect(lambda: QApplication.clipboard().setText(self.log_view.toPlainText()))
        right.addWidget(self.log_view)
        tr = QHBoxLayout()
        right.addLayout(tr)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedWidth(100)
        self.play_btn.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)
        self.frame_lbl = QLabel("Frame -")
        self.frame_lbl.setMinimumWidth(170)
        still = QPushButton("📸 Save this frame")
        still.setObjectName("ghost")
        still.clicked.connect(self.save_still)
        for wdg in (self.play_btn, self.slider, self.frame_lbl, still):
            tr.addWidget(wdg)

        self.timer = QTimer(singleShot=True, interval=150)
        self.timer.timeout.connect(self.request_preview)
        self.play_timer = QTimer(interval=15)
        self.play_timer.timeout.connect(self.play_tick)
        for s in (self.gamma, self.smooth, self.black, self.white, self.clean, self.nstr, self.blend):
            s.valueChanged.connect(lambda *_: self.timer.start())
        for c in (self.invert, self.tta, self.prev_full, self.nflip):
            c.stateChanged.connect(lambda *_: self.timer.start())
        for c in (self.cmap, self.detail, self.mode):
            c.currentIndexChanged.connect(lambda *_: self.timer.start())
        self.setStyleSheet(STYLE)
        log("Depth Studio v%s started | onnxruntime %s | backend: %s | providers: %s" % (VERSION, ort.__version__, BACKEND or "CPU only", ", ".join(_AV)))
        log("Models found: %s" % (", ".join(os.path.basename(p) for p in self.models.values()) or "none"))
        QTimer.singleShot(200, self.autoload)

        # update check (background, prints debug output to console)
        self.updater = Updater()
        self.updater.found.connect(self.on_update)
        QTimer.singleShot(1500, self.updater.start)

    def on_update(self, tag, notes, url):
        log("Update available: %s (you have v%s)" % (tag, VERSION))
        m = QMessageBox.question(self, "Update available",
                                 "Depth Studio %s is available (you have v%s).\n\n%s\n\nOpen the download page?"
                                 % (tag, VERSION, notes))
        if m == QMessageBox.StandardButton.Yes and url:
            webbrowser.open(url)

    def autoload(self):
        name = CFG.get("model")
        if name not in self.models:
            name = list(self.models)[0] if self.models else None
        if name:
            self.model_cb.setCurrentText(name)
            self.start_load(name, self.models[name])
        else:
            self.status.setText("No model found. Put the .onnx files in a 'models' folder next to the program, "
                                "or choose 'Browse for .onnx file...'.")

    def start_load(self, name, path):
        if self.exporting:
            return
        if self.loader and self.loader.isRunning():
            self._pending_load = (name, path)
            return
        self.cur_model = name
        self.status.setText("Loading %s (first load can take a minute)..." % name)
        self.loader = LoadWorker(self.engine, path)
        self.loader.done.connect(self.on_loaded)
        self.loader.failed.connect(lambda m: self.status.setText(m))
        self.loader.start()

    def on_loaded(self, path):
        CFG["model"] = self.cur_model
        cfg_save(CFG)
        eng = self.engine
        self.dev_lbl.setText("Device: " + eng.describe())
        self.detail.setEnabled(eng.fixed is None)
        self.tta.setEnabled(not eng.is_video)
        self.outtype.setCurrentIndex(eng.out_mode)
        guess = "depth (far = bright, converted)" if eng.depth_out() else "disparity (near = bright)"
        self.status.setText("Model ready: %s  -  treated as %s. If it looks inverted, change 'Model output'." %
                            (os.path.basename(path), guess))
        if self._pending_load:
            n, p = self._pending_load
            self._pending_load = None
            return self.start_load(n, p)
        self.request_preview()

    def model_chosen(self, idx):
        text = self.model_cb.itemText(idx)
        if text == BROWSE:
            p, _ = QFileDialog.getOpenFileName(self, "Choose model", APP_DIR, "ONNX model (*.onnx)")
            text = self.add_model(p) if p else None
            if not text:
                return self.model_cb.setCurrentText(self.cur_model or "")
        self.start_load(text, self.models[text])

    def add_model(self, p):
        stem = os.path.splitext(os.path.basename(p))[0]
        if is_vda_head(stem):
            QMessageBox.information(self, "Depth Studio", "That is the temporal head of a Video Depth Anything pair.\n"
                                    "Choose the matching *_encoder_* file instead (keep both files in the same folder).")
            return None
        name = pretty(stem)
        if name in self.models and self.models[name] != p:
            name = "%s [%s]" % (name, stem)
        self.models[name] = p
        self.models = dict(sorted(self.models.items(),
                                  key=lambda kv: sort_key(os.path.splitext(os.path.basename(kv[1]))[0])))
        self.model_cb.clear()
        self.model_cb.addItems(list(self.models) + [BROWSE])
        self.model_cb.setCurrentText(name)
        self.refresh_model2()
        return name

    def refresh_model2(self):
        cur = self.model2_cb.currentText()
        self.model2_cb.blockSignals(True)
        self.model2_cb.clear()
        self.model2_cb.addItem(NO_BLEND)
        for n, p in self.models.items():
            if not is_vda_enc(os.path.splitext(os.path.basename(p))[0]):
                self.model2_cb.addItem(n)
        self.model2_cb.setCurrentIndex(max(0, self.model2_cb.findText(cur)))
        self.model2_cb.blockSignals(False)

    def model2_chosen(self, idx):
        text = self.model2_cb.itemText(idx)
        if self.exporting:
            return
        if text == NO_BLEND:
            self.engine2.unload()
            self.status.setText("Detail model off.")
            return self.request_preview()
        self.start_load2(text, self.models[text])

    def start_load2(self, name, path):
        if self.exporting:
            return
        if self.loader2 and self.loader2.isRunning():
            self._pending2 = (name, path)
            return
        self.status.setText("Loading detail model %s..." % name)
        self.loader2 = LoadWorker(self.engine2, path)
        self.loader2.done.connect(self.on_loaded2)
        self.loader2.failed.connect(lambda m: (self.status.setText(m), self.model2_cb.setCurrentIndex(0)))
        self.loader2.start()

    def on_loaded2(self, path):
        self.status.setText("Detail model ready: %s  (needs extra GPU memory; lower Detail level if you run out)" % os.path.basename(path))
        if self._pending2:
            n, p = self._pending2
            self._pending2 = None
            return self.start_load2(n, p)
        self.request_preview()

    def outtype_chosen(self, idx):
        if self.exporting:
            return
        self.engine.out_mode = idx
        log("Model output set to: %s" % OUT_TYPES[idx])
        self.request_preview()

    def gpu_chosen(self, idx):
        global GPU_ID
        if self.exporting:
            return
        GPU_ID = idx
        CFG["gpu_id"] = idx
        cfg_save(CFG)
        if self.engine.path and self.cur_model:
            self.start_load(self.cur_model, self.engine.path)
        if self.engine2.path and self.model2_cb.currentIndex() > 0:
            self.start_load2(self.model2_cb.currentText(), self.engine2.path)

    def settings(self):
        return dict(size=self.detail.currentData(), cmap=self.cmap.currentText(), gamma=self.gamma.value() / 100,
                    smooth=self.smooth.value() / 100, black=self.black.value() / 100, white=self.white.value() / 100,
                    clean=self.clean.value(), invert=self.invert.isChecked(), tta=self.tta.isChecked(),
                    blend=self.blend.value() / 100 if self.model2_cb.currentIndex() > 0 else 0.,
                    mode=self.mode.currentIndex(), nstrength=self.nstr.value() / 100, nflip=self.nflip.isChecked(),
                    format=self.fmt.currentText(), quality=self.quality.currentIndex(), audio=self.audio.isChecked())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        vids = [p for p in paths if p.lower().endswith(VIDEO_EXT)]
        for p in paths:
            if p.lower().endswith(".onnx"):
                name = self.add_model(p)
                if name:
                    self.start_load(name, p)
        if vids:
            self.load_video(vids[0])
            self.queue_add(vids[1:])

    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.wmv)")
        if p:
            self.load_video(p)

    def load_video(self, p):
        if self.exporting:
            return
        self.stop_play()
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            return QMessageBox.warning(self, "Depth Studio", "Couldn't open that video.")
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), cap.get(cv2.CAP_PROP_FPS) or 24.0
        w, h = int(cap.get(3)), int(cap.get(4))
        if n <= 0:
            while cap.grab():
                n += 1
        cap.release()
        if n <= 0 or not w:
            return QMessageBox.warning(self, "Depth Studio", "Couldn't read any frames from that video.")
        self.video, self.nframes, self.fps = p, n, fps
        self.info.setText("%s\n%dx%d  %.2f fps  %d frames" % (os.path.basename(p), w, h, fps, n))
        log("Opened video: %s (%dx%d, %.2f fps, %d frames)" % (p, w, h, fps, n))
        for sp in (self.start_spin, self.end_spin):
            sp.setRange(0, n - 1)
        self.start_spin.setValue(0)
        self.end_spin.setValue(n - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.set_frame_label(0)
        self.request_preview()

    def queue_add(self, paths):
        have = {self.queue.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.queue.count())}
        for p in paths:
            if p not in have:
                it = QListWidgetItem(os.path.basename(p))
                it.setData(Qt.ItemDataRole.UserRole, p)
                self.queue.addItem(it)

    def queue_add_dialog(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.wmv)")
        self.queue_add(ps)

    def queue_add_current(self):
        if self.video:
            self.queue_add([self.video])

    def queue_remove(self):
        for it in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(it))

    def show_img(self, lbl, rgb):
        self._imgs[lbl] = rgb
        lbl.setPixmap(to_pixmap(rgb).scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for lbl, rgb in list(self._imgs.items()):
            self.show_img(lbl, rgb)

    def set_frame_label(self, i):
        self.frame_lbl.setText("Frame %d / %d  -  %.2fs" % (i, self.nframes - 1, i / self.fps))

    def on_slider(self, i):
        self.set_frame_label(i)
        self.request_preview()

    def request_preview(self):
        if not self.video or self.exporting:
            return
        s = self.settings()
        full = self.prev_full.isChecked()
        self.pw.request(dict(video=self.video, idx=self.slider.value(), s=s,
                             size=s["size"] if full else min(s["size"], 518), tta=bool(s["tta"] and full)))

    def on_ready(self, idx, src, dep, n, info):
        if self.exporting or (not self.playing and idx != self.slider.value()):
            return
        self.show_img(self.src_lbl, src)
        if dep is not None:
            self.show_img(self.dep_lbl, dep)
            self.cur_n, self.cur_idx = n, idx
        self.perf.setText(info)

    def toggle_play(self):
        if self.playing:
            return self.stop_play()
        if not self.video or self.exporting:
            return
        s, e = self.start_spin.value(), self.end_spin.value()
        if not (s <= self.slider.value() < e):
            self.slider.setValue(s)
        self.playing = True
        self.play_btn.setText("⏸ Pause")
        self.play_i0, self.play_t0 = self.slider.value(), time.perf_counter()
        self.play_timer.start()

    def stop_play(self):
        self.playing = False
        self.play_timer.stop()
        self.play_btn.setText("▶ Play")

    def play_tick(self):
        if not self.playing:
            return
        s, e = self.start_spin.value(), self.end_spin.value()
        i = self.play_i0 + int((time.perf_counter() - self.play_t0) * self.fps)
        if i > e:
            self.play_i0, self.play_t0, i = s, time.perf_counter(), s
        if i != self.slider.value():
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)
            self.set_frame_label(i)
            self.request_preview()

    def save_still(self):
        if not self.video or not self.engine.ready():
            return QMessageBox.information(self, "Depth Studio", "Open a video and let the model load first.")
        if self.still and self.still.isRunning():
            return
        i = self.slider.value()
        tag = "normal" if self.mode.currentIndex() == 1 else "depth"
        base = os.path.splitext(self.video)[0] + "_%s_%06d" % (tag, i)
        p, _ = QFileDialog.getSaveFileName(self, "Save frame", base + ".png", "PNG (*.png);;EXR (*.exr)")
        if not p:
            return
        self.status.setText("Saving frame at full quality...")
        self.still = StillWorker(self.engine, self.video, i, self.settings(), p, self.engine2)
        self.still.done.connect(lambda m: self.status.setText(m))
        self.still.start()

    def set_busy(self, busy):
        self.exporting = busy
        self.pw.paused = busy
        self.export_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def export(self):
        if not self.engine.ready():
            return QMessageBox.information(self, "Depth Studio", "No model loaded yet. Choose one under AI model.")
        if (self.loader and self.loader.isRunning()) or (self.loader2 and self.loader2.isRunning()):
            return QMessageBox.information(self, "Depth Studio", "A model is still loading. Try again in a moment.")
        if self.queue.count():
            jobs = [{"video": self.queue.item(i).data(Qt.ItemDataRole.UserRole), "start": 0, "end": -1}
                    for i in range(self.queue.count())]
        elif self.video:
            jobs = [{"video": self.video, "start": self.start_spin.value(), "end": self.end_spin.value()}]
        else:
            return QMessageBox.information(self, "Depth Studio", "Open a video first.")
        out = QFileDialog.getExistingDirectory(self, "Choose output folder", self.last_out or "")
        if not out:
            return
        s = self.settings()
        s["out_dir"] = out
        self.stop_play()
        self.last_out = out
        CFG["out_dir"] = out
        cfg_save(CFG)
        self.bar.setValue(0)
        self.set_busy(True)
        log("EXPORT started: %d clip(s) | format: %s | mode: %s | model: %s | detail model: %s (blend %d%%) | detail level: %s | "
            "noise reduction: %s | out: %s" % (len(jobs), s["format"], "normal map" if s["mode"] == 1 else "depth map",
                                               self.cur_model, self.model2_cb.currentText(), round(s["blend"] * 100),
                                               "model's fixed size" if self.engine.fixed else s["size"], s["tta"], out))
        if self.engine.is_video:
            self.status.setText("Video model: the first frames appear after the first %d-frame window is processed..." % self.engine.T)
        self.worker = ExportWorker(s, jobs, self.engine, self.engine2)
        self.worker.progress.connect(lambda p, t: (self.bar.setValue(p), self.status.setText(t)))
        self.worker.preview.connect(lambda a, b: (self.show_img(self.src_lbl, a), self.show_img(self.dep_lbl, b)))
        self.worker.finished_ok.connect(self.export_done)
        self.worker.failed.connect(self.export_failed)
        self.worker.start()

    def cancel_export(self):
        if self.worker:
            self.worker.cancel = True
            self.status.setText("Cancelling...")

    def export_done(self, out, cancelled):
        self.set_busy(False)
        self.open_btn.setEnabled(True)
        self.status.setText("Cancelled." if cancelled else "Done! Saved to: %s" % out)
        if not cancelled:
            self.bar.setValue(100)
        self.request_preview()

    def export_failed(self, err):
        self.set_busy(False)
        self.status.setText("Export failed.")
        log("EXPORT FAILED:\n" + err[-3000:])
        QMessageBox.critical(self, "Error", err[-1500:])
        self.request_preview()

    def open_folder(self):
        if not self.last_out:
            return
        if sys.platform == "win32":
            os.startfile(self.last_out)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.last_out])
        else:
            subprocess.Popen(["xdg-open", self.last_out])

    def closeEvent(self, e):
        self.stop_play()
        if self.worker and self.worker.isRunning():
            self.worker.cancel = True
            self.worker.wait(8000)
        self.pw.stop()
        self.pw.wait(3000)
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    sys.excepthook = lambda t, v, tb: log("UNCAUGHT ERROR:\n" + "".join(traceback.format_exception(t, v, tb)))
    threading.excepthook = lambda a: log("ERROR in thread %s:\n%s" % (a.thread.name if a.thread else "?", "".join(
        traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback))))
    win = Main()
    win.show()
    sys.exit(app.exec())