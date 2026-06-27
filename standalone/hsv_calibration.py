"""
HSVマスクキャリブレーションツール
4カメラそれぞれのHSV設定をGUIで調整し、カメラ別JSONファイルとして保存する。

保存先: hsv_config_{cam_name}.json
  例: hsv_config_cam_top.json, hsv_config_cam_under.json ...

スライダー構成（カメラごとに独立）:
  H1最小/最大 … 赤色範囲1の色相
  H2最小/最大 … 赤色範囲2の色相
  S 最小/最大 … 彩度（H1/H2 共通）
  V 最小/最大 … 明度（H1/H2 共通）

検出条件:
  モルフォロジー処理後のマスク面積が MIN_AREA 以上の成分のみ有効とする。
"""

import sys
import json
import os
import threading
import cv2
import numpy as np
from pypylon import pylon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QSpinBox,
    QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# ==========================================================
# 設定
# ==========================================================
TARGET_SERIALS = [
    ("25453227", "cam_top"),
    ("25453229", "cam_under"),
    ("25308967", "cam_inside"),
    ("25308968", "cam_outside"),
]

MIN_AREA = 1000          # 最小検出面積のデフォルト値（パネルで個別調整可）
PREVIEW_W, PREVIEW_H = 360, 360   # プレビュー表示サイズ（4画面一覧用）

# グリッド配置: (cam_name, row, col)  ← ここを変えるだけで並びを変更できる
GRID_LAYOUT = [
    ("cam_inside",  0, 0),   # 左上
    ("cam_outside", 0, 1),   # 右上
    ("cam_under",   1, 0),   # 左下
    ("cam_top",     1, 1),   # 右下
]

# デフォルト HSV 値（既存 hsv_config.json と同じ初期値）
DEFAULT_HSV = {
    "lower1": [0,   30, 47],
    "upper1": [37,  255, 255],
    "lower2": [160, 30, 47],
    "upper2": [180, 255, 255],
}

# スライダー定義: (表示名, JSONキー, インデックス, 最大値)
# ※ S/V は lower1/upper1 のインデックス1,2 に書き込み、lower2/upper2 にも同値をコピーする
SLIDER_DEFS = [
    ("H1 最小", "lower1", 0, 180),
    ("H1 最大", "upper1", 0, 180),
    ("H2 最小", "lower2", 0, 180),
    ("H2 最大", "upper2", 0, 180),
    ("S  最小", "lower1", 1, 255),
    ("S  最大", "upper1", 1, 255),
    ("V  最小", "lower1", 2, 255),
    ("V  最大", "upper1", 2, 255),
]


# ==========================================================
# JSON 入出力
# ==========================================================
HSV_JSON_DIR = "json"


def config_path(cam_name: str) -> str:
    return os.path.join(HSV_JSON_DIR, f"hsv_config_{cam_name}.json")


def load_config(cam_name: str) -> dict:
    """カメラ別JSON → 共通JSON → デフォルト値 の順にフォールバック。"""
    common = os.path.join(HSV_JSON_DIR, "hsv_common_config.json")
    for path in (config_path(cam_name), common):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {k: list(v) for k, v in DEFAULT_HSV.items()}


def save_config(cam_name: str, params: dict) -> str:
    path = config_path(cam_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    return path


# ==========================================================
# 画像処理
# ==========================================================
def apply_mask(frame: np.ndarray, params: dict) -> np.ndarray:
    """HSVマスクを生成して返す（module_yolo_csv4.py の get_target_info と同一処理）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(params["lower1"]), np.array(params["upper1"])),
        cv2.inRange(hsv, np.array(params["lower2"]), np.array(params["upper2"])),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def draw_overlay(frame: np.ndarray, mask: np.ndarray, min_area: int = MIN_AREA) -> tuple[np.ndarray, int]:
    """min_area 以上の有効領域を緑枠で描画した overlay と最大面積を返す。"""
    overlay = frame.copy()
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    max_area = 0
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(overlay, f"{area:,}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        max_area = max(max_area, area)
    return overlay, max_area


def to_pixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


# ==========================================================
# カメラキャプチャスレッド
# ==========================================================
class CameraThread:
    def __init__(self, serial: str, cam_name: str):
        self.serial = serial
        self.cam_name = cam_name
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_flag = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop_flag = True

    def _run(self):
        try:
            tl = pylon.TlFactory.GetInstance()
            device = next(
                (d for d in tl.EnumerateDevices() if d.GetSerialNumber() == self.serial),
                None,
            )
            if device is None:
                print(f"[{self.cam_name}] カメラが見つかりません (serial={self.serial})")
                return

            cam = pylon.InstantCamera(tl.CreateDevice(device))
            cam.Open()

            pfs_path = f"cam_pfs/{self.cam_name}_{self.serial}.pfs"
            if os.path.exists(pfs_path):
                pylon.FeaturePersistence.Load(pfs_path, cam.GetNodeMap(), True)

            conv = pylon.ImageFormatConverter()
            conv.OutputPixelFormat = pylon.PixelType_BGR8packed
            conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self._stop_flag and cam.IsGrabbing():
                result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if result.GrabSucceeded():
                    with self._lock:
                        self._frame = conv.Convert(result).GetArray().copy()
                result.Release()
            cam.StopGrabbing()
            cam.Close()
        except Exception as e:
            print(f"[{self.cam_name}] エラー: {e}")


# ==========================================================
# 1カメラ分のスライダーパネル
# ==========================================================
class ParamPanel(QWidget):
    def __init__(self, cam_name: str):
        super().__init__()
        self.cam_name = cam_name
        self._params = load_config(cam_name)
        self._sliders: list[QSlider] = []
        self._spinboxes: list[QSpinBox] = []
        self._build_ui()
        self._params_to_sliders()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # HSV スライダー＋数値入力
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        for row, (name, _, _, max_val) in enumerate(SLIDER_DEFS):
            lbl_name = QLabel(name)
            lbl_name.setFixedWidth(65)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, max_val)
            slider.setFixedHeight(18)

            spin = QSpinBox()
            spin.setRange(0, max_val)
            spin.setFixedWidth(55)
            spin.setButtonSymbols(QSpinBox.NoButtons)

            # 双方向同期（同値の場合は valueChanged が再発火しないため無限ループにならない）
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(self._on_change)

            self._sliders.append(slider)
            self._spinboxes.append(spin)

            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(slider,   row, 1)
            grid.addWidget(spin,     row, 2)

        layout.addLayout(grid)

        # 最小面積スライダー＋数値入力（0 〜 200,000）
        area_row = QHBoxLayout()
        area_lbl = QLabel("最小面積")
        area_lbl.setFixedWidth(65)
        self._area_slider = QSlider(Qt.Horizontal)
        self._area_slider.setRange(0, 200000)
        self._area_slider.setValue(MIN_AREA)
        self._area_slider.setFixedHeight(18)
        self._area_spin = QSpinBox()
        self._area_spin.setRange(0, 200000)
        self._area_spin.setValue(MIN_AREA)
        self._area_spin.setFixedWidth(75)
        self._area_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._area_slider.valueChanged.connect(self._area_spin.setValue)
        self._area_spin.valueChanged.connect(self._area_slider.setValue)
        area_row.addWidget(area_lbl)
        area_row.addWidget(self._area_slider)
        area_row.addWidget(self._area_spin)
        layout.addLayout(area_row)

        btn = QPushButton(f"保存: {self.cam_name}")
        btn.clicked.connect(self._save)
        layout.addWidget(btn)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def _params_to_sliders(self):
        """_params の値をスライダー・スピンボックスに反映する。"""
        for i, (_, key, idx, _) in enumerate(SLIDER_DEFS):
            val = self._params[key][idx]
            self._sliders[i].blockSignals(True)
            self._spinboxes[i].blockSignals(True)
            self._sliders[i].setValue(val)
            self._spinboxes[i].setValue(val)
            self._sliders[i].blockSignals(False)
            self._spinboxes[i].blockSignals(False)
        self._on_change()

    def _on_change(self):
        h1_min = self._sliders[0].value()
        h1_max = self._sliders[1].value()
        h2_min = self._sliders[2].value()
        h2_max = self._sliders[3].value()
        s_min  = self._sliders[4].value()
        s_max  = self._sliders[5].value()
        v_min  = self._sliders[6].value()
        v_max  = self._sliders[7].value()
        self._params = {
            "lower1": [h1_min, s_min, v_min],
            "upper1": [h1_max, s_max, v_max],
            "lower2": [h2_min, s_min, v_min],
            "upper2": [h2_max, s_max, v_max],
        }

    def get_params(self) -> dict:
        return self._params

    def get_min_area(self) -> int:
        return self._area_spin.value()

    def _save(self):
        path = save_config(self.cam_name, self._params)
        self._status.setText(f"保存完了:\n{path}")


# ==========================================================
# メインウィンドウ（4カメラ 2×2 一覧表示）
# ==========================================================
class CalibWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV キャリブレーション（4カメラ一覧）")
        self.resize(1300, 920)

        self._threads: dict[str, CameraThread] = {}
        self._cam_ui: dict[str, dict] = {}

        self._build_ui()
        self._start_cameras()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20 fps

    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        root.addWidget(grid_widget)

        for cam_name, row, col in GRID_LAYOUT:
            self._add_camera_cell(cam_name, grid_layout, row, col)

        btn_all = QPushButton("▼ 全カメラを一括保存")
        btn_all.setFixedHeight(38)
        btn_all.clicked.connect(self._save_all)
        root.addWidget(btn_all)

        self._sb = self.statusBar()
        self._sb.showMessage("カメラ接続中...")

    def _add_camera_cell(self, cam_name: str, grid_layout: QGridLayout, row: int, col: int):
        grp = QGroupBox(cam_name)
        cell_layout = QHBoxLayout(grp)
        cell_layout.setContentsMargins(4, 4, 4, 4)
        cell_layout.setSpacing(8)

        # マスク後画像 ＋ 検出情報
        img_v = QVBoxLayout()
        mask_lbl = self._make_preview_label()
        img_v.addWidget(mask_lbl)

        info_lbl = QLabel("面積: --\n検出: --")
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setFixedHeight(40)
        info_lbl.setStyleSheet("font-size: 13px;")
        img_v.addWidget(info_lbl)
        img_v.addStretch()

        cell_layout.addLayout(img_v)

        # スライダーパネル
        panel = ParamPanel(cam_name)
        panel.setFixedWidth(265)
        cell_layout.addWidget(panel)

        grid_layout.addWidget(grp, row, col)

        self._cam_ui[cam_name] = {
            "mask":  mask_lbl,
            "info":  info_lbl,
            "panel": panel,
        }

    @staticmethod
    def _make_preview_label() -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(PREVIEW_W, PREVIEW_H)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background:#1a1a1a;")
        return lbl

    # --------------------------------------------------
    def _start_cameras(self):
        for serial, cam_name in TARGET_SERIALS:
            t = CameraThread(serial, cam_name)
            t.start()
            self._threads[cam_name] = t
        self._sb.showMessage("カメラ起動中（映像が表示されるまで少しお待ちください）")

    # --------------------------------------------------
    def _tick(self):
        """タイマーで定期的に全カメラのマスク後プレビューを更新する。"""
        for cam_name, _, _ in GRID_LAYOUT:
            ui = self._cam_ui[cam_name]
            frame = self._threads[cam_name].get_frame()
            if frame is None:
                continue

            params = ui["panel"].get_params()
            min_area = ui["panel"].get_min_area()
            mask = apply_mask(frame, params)
            overlay, max_area = draw_overlay(frame, mask, min_area)
            ui["mask"].setPixmap(to_pixmap(overlay, PREVIEW_W, PREVIEW_H))

            if max_area >= min_area:
                ui["info"].setText(f"最大面積: {max_area:,}\n検出: あり ✓")
                ui["info"].setStyleSheet("color:#00cc44; font-size:13px; font-weight:bold;")
            else:
                area_txt = f"{max_area:,}" if max_area > 0 else "--"
                ui["info"].setText(f"最大面積: {area_txt}\n検出: なし")
                ui["info"].setStyleSheet("color:#cc4444; font-size:13px;")

    # --------------------------------------------------
    def _save_all(self):
        saved = []
        for cam_name, _, _ in GRID_LAYOUT:
            params = self._cam_ui[cam_name]["panel"].get_params()
            save_config(cam_name, params)
            saved.append(cam_name)
        self._sb.showMessage(f"全カメラ保存完了: {', '.join(saved)}")

    def closeEvent(self, event):
        self._timer.stop()
        for t in self._threads.values():
            t.stop()
        event.accept()


# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CalibWindow()
    win.show()
    sys.exit(app.exec())
