"""GUI表示経路のコスト比較: PySide6(現状) vs Tkinter+PIL。
表示先ラベルサイズは 480x360 相当と仮定（4分割プレビュー）。"""
import os, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np, cv2

N = 200
SIZES = [640, 640, 560, 500]
DST = (480, 360)

def bench(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0

frames = [np.random.randint(0, 255, (s, s, 3), dtype=np.uint8) for s in SIZES]

# ---------- PySide6 ----------
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
app = QApplication([])

def qt_path(frame, mode):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(DST[0], DST[1], Qt.KeepAspectRatio, mode)

tot_smooth = sum(bench(lambda f=f: qt_path(f, Qt.SmoothTransformation)) for f in frames)
tot_fast   = sum(bench(lambda f=f: qt_path(f, Qt.FastTransformation)) for f in frames)

# cv2でリサイズしてからQImage化（推奨案）
def qt_cv_resize(frame):
    small = cv2.resize(frame, DST, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

tot_cvresize = sum(bench(lambda f=f: qt_cv_resize(f)) for f in frames)

print("=== PySide6 (4台合計 / 1ティックあたり) ===")
print(f"  現状 SmoothTransformation : {tot_smooth:6.2f} ms")
print(f"  FastTransformation        : {tot_fast:6.2f} ms")
print(f"  cv2.resize後にQImage化     : {tot_cvresize:6.2f} ms  <-- 推奨")
print()

# ---------- Tkinter + PIL ----------
try:
    from PIL import Image, ImageTk
    import tkinter as tk
    root = tk.Tk(); root.withdraw()

    def tk_path(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize(DST, Image.BILINEAR)
        return ImageTk.PhotoImage(img)

    tot_tk = sum(bench(lambda f=f: tk_path(f)) for f in frames)

    def tk_path_cvresize(frame):
        small = cv2.resize(frame, DST, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    tot_tk2 = sum(bench(lambda f=f: tk_path_cvresize(f)) for f in frames)

    print("=== Tkinter + PIL (4台合計 / 1ティックあたり) ===")
    print(f"  PIL resize + ImageTk      : {tot_tk:6.2f} ms")
    print(f"  cv2.resize + ImageTk      : {tot_tk2:6.2f} ms")
except Exception as e:
    print("Tkinter/PIL 計測不可:", e)
