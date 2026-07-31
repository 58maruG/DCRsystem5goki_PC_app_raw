"""
HSVカラービューア（OpenCV基準）

スライダーまたは数値ボックスでHSVを変えると、その色をGUIに即表示するだけの確認ツール。
値の解釈は OpenCV に厳密準拠（H:0-179 / S:0-255 / V:0-255）。
表示色は cv2.cvtColor(HSV2BGR) で変換するので、コード側のマスク値と見え方が完全に一致する。

補助表示:
  ・RGB / HEX（画面の色）
  ・一般的なHSV（H:0-360 / S:0-100 / V:0-100）… 基準紙やカラーピッカーとの対応確認用

起動:
  uv run python standalone/hsv_color_viewer.py
"""

import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QSpinBox,
    QGridLayout, QVBoxLayout, QHBoxLayout, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# チャンネル定義: (表示名, 最大値)  ※OpenCV基準
CHANNELS = [
    ("H", 179),   # 色相 0-179（180は0と同じ）
    ("S", 255),   # 彩度 0-255
    ("V", 255),   # 明度 0-255
]

INIT_HSV = (0, 255, 255)   # 初期値（純赤）


def hsv_to_rgb(h: int, s: int, v: int) -> tuple[int, int, int]:
    """OpenCVのHSV(0-179/0-255/0-255)を、OpenCVの解釈どおりRGBへ変換する。"""
    px  = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])   # R, G, B


class HsvChannel(QWidget):
    """1チャンネル分の「ラベル + スライダー + 数値ボックス」。両者は相互連動する。"""

    def __init__(self, name: str, max_val: int, init: int, on_change):
        super().__init__()
        self._on_change = on_change

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(name)
        lbl.setFixedWidth(24)
        lbl.setFont(QFont("", 12, QFont.Bold))
        row.addWidget(lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max_val)
        self.slider.setValue(init)
        row.addWidget(self.slider, 1)

        self.spin = QSpinBox()
        self.spin.setRange(0, max_val)
        self.spin.setValue(init)
        self.spin.setFixedWidth(70)
        self.spin.setFont(QFont("", 11))
        row.addWidget(self.spin)

        rng = QLabel(f"/ {max_val}")
        rng.setFixedWidth(44)
        rng.setStyleSheet("color:#888;")
        row.addWidget(rng)

        # スライダー ↔ 数値ボックスの相互連動（無限ループ防止に signal を一時停止）
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, val: int):
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self._on_change()

    def _from_spin(self, val: int):
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self._on_change()

    def value(self) -> int:
        return self.slider.value()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSVカラービューア（OpenCV基準）")
        self.resize(560, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # --- 色見本 ---
        self.swatch = QFrame()
        self.swatch.setMinimumHeight(200)
        self.swatch.setFrameShape(QFrame.StyledPanel)
        root.addWidget(self.swatch, 1)

        # --- スライダー群 ---
        self._channels: list[HsvChannel] = []
        for name, max_val in CHANNELS:
            init = INIT_HSV[len(self._channels)]
            ch = HsvChannel(name, max_val, init, self._update)
            self._channels.append(ch)
            root.addWidget(ch)

        # --- 数値表示（HSV / RGB / HEX / 一般HSV）---
        info = QGridLayout()
        info.setHorizontalSpacing(10)
        info.setVerticalSpacing(4)
        self._val_labels: dict[str, QLabel] = {}
        rows = [
            ("OpenCV HSV", "hsv"),
            ("RGB",        "rgb"),
            ("HEX",        "hex"),
            ("一般HSV (0-360/0-100)", "hsv360"),
        ]
        for r, (title, key) in enumerate(rows):
            t = QLabel(title)
            t.setStyleSheet("color:#666;")
            v = QLabel("-")
            v.setFont(QFont("Consolas", 12, QFont.Bold))
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)   # コピー可
            info.addWidget(t, r, 0, Qt.AlignRight)
            info.addWidget(v, r, 1, Qt.AlignLeft)
            self._val_labels[key] = v
        root.addLayout(info)

        self._update()

    def _update(self):
        h, s, v = (c.value() for c in self._channels)
        r, g, b = hsv_to_rgb(h, s, v)

        # 色見本を更新
        self.swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border:1px solid #999;"
        )

        # 数値表示を更新
        self._val_labels["hsv"].setText(f"H={h}  S={s}  V={v}")
        self._val_labels["rgb"].setText(f"R={r}  G={g}  B={b}")
        self._val_labels["hex"].setText(f"#{r:02X}{g:02X}{b:02X}")
        # 一般的なHSV表記（基準紙・カラーピッカーと対応）
        h360 = round(h * 2)
        s100 = round(s / 255 * 100)
        v100 = round(v / 255 * 100)
        self._val_labels["hsv360"].setText(f"H={h360}°  S={s100}%  V={v100}%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
