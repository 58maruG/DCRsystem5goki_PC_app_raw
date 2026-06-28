"""
HSVフィルタ保存ツール（クラスごと設定・画像フィルタ保存）

train用_v4 フォルダの各クラス画像にHSVマスクを適用し、
クラスごとに個別の HSV パラメータを設定・保存した上で、
マスクで検出された画像のみを別フォルダに出力するGUI。

使い方:
  1. 左の「クラス一覧」でクラスをクリックして選択する。
  2. スライダーで HSV パラメータを調整する（緑枠=検出成功/赤枠=失敗）。
  3. 「現クラスHSV設定を保存」でそのクラスの JSON を書き出す。
  4. 「検出画像をコピー保存」または「検出画像をクロップ保存」で
     マスク通過画像を別フォルダに出力する。

保存先:
  コピー : train用_v4_hsv_filtered/{class_name}/  ← 元画像をそのままコピー
  クロップ: train用_v4_hsv_cropped/{class_name}/   ← 検出領域のみ切り出し保存

面積閾値について:
  スライダーはサムネイル(THUMB_SIZE=112px)換算の面積で設定する。
  保存処理時は元画像スケールに自動換算して再検出する。
"""

import sys
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton,
    QScrollArea, QListWidget, QListWidgetItem,
    QGridLayout, QGroupBox,
    QProgressDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont

# ==========================================================
# 設定
# ==========================================================
TRAIN_DIR   = r"C:\Users\kotan\gohara\cherry_yolo\model作成用imageset\all\healthy"
THUMB_SIZE  = 112
GRID_COLS   = 7
DEBOUNCE_MS = 300

_TRAIN_PATH  = Path(TRAIN_DIR)
FILTERED_DIR = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_hsv_filtered"
CROPPED_DIR  = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_hsv_cropped"

# standalone の親 = DCRsystem5goki_PC_app_raw 配下の json/ を共有
_JSON_DIR           = Path(__file__).parent.parent / "json"
_COMMON_CONFIG_PATH = _JSON_DIR / "hsv_common_config.json"

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

DEFAULT_HSV = {
    "lower1": [0,   30, 47],
    "upper1": [39,  255, 255],
    "lower2": [160, 30, 47],
    "upper2": [180, 255, 255],
}

DEFAULT_MIN_AREA        = 500
DEFAULT_MAX_AREA        = 0
DEFAULT_MIN_CIRCULARITY = 0.0


# ==========================================================
# JSON 入出力
# ==========================================================
def _class_config_path(cls_name: str) -> Path:
    return _JSON_DIR / f"hsv_class_{cls_name}.json"


def load_class_config(cls_name: str) -> dict:
    """クラス別 JSON を読む。なければ共通設定、それもなければデフォルト値。"""
    p = _class_config_path(cls_name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if _COMMON_CONFIG_PATH.exists():
        with open(_COMMON_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {k: list(v) for k, v in DEFAULT_HSV.items()}


def save_class_config(cls_name: str, params: dict):
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(_class_config_path(cls_name), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)


def has_class_config(cls_name: str) -> bool:
    return _class_config_path(cls_name).exists()


# ==========================================================
# 日本語パス対応の imread / imwrite
# ==========================================================
def imread_unicode(path: str) -> np.ndarray | None:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    try:
        ext = Path(path).suffix.lower()
        ret, buf = cv2.imencode(ext, img)
        if not ret:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# ==========================================================
# 画像処理
# ==========================================================
def compute_mask(frame: np.ndarray, params: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(params["lower1"]), np.array(params["upper1"]))
    m2 = cv2.inRange(hsv, np.array(params["lower2"]), np.array(params["upper2"]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def get_detections(
    frame: np.ndarray,
    params: dict,
    min_area: int,
    max_area: int = 0,
    min_circularity: float = 0.0,
) -> list:
    """HSVマスク処理後の有効領域の矩形リストを返す。"""
    mask = compute_mask(frame, params)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if min_circularity > 0.0:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append((x, y, w, h))
    return rects


def get_detections_fullsize(
    img: np.ndarray,
    params: dict,
    thumb_min_area: int,
    thumb_max_area: int,
    min_circularity: float,
) -> list:
    """元画像スケールで検出する。面積閾値はサムネイル換算値から自動変換する。"""
    h, w = img.shape[:2]
    scale = min(THUMB_SIZE / w, THUMB_SIZE / h)
    area_factor = 1.0 / (scale ** 2)
    min_area = max(1, int(thumb_min_area * area_factor))
    max_area = int(thumb_max_area * area_factor) if thumb_max_area > 0 else 0
    return get_detections(img, params, min_area, max_area, min_circularity)


def make_thumb_pixmap(thumb: np.ndarray, size: int, rects: list) -> QPixmap:
    """サムネイルに検出矩形を描画して QPixmap に変換する。"""
    ok = len(rects) > 0
    annotated = thumb.copy()
    for (rx, ry, rw, rh) in rects:
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 0), 2)
    h, w = annotated.shape[:2]
    scale = min(size / w, size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(annotated, (nw, nh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data.tobytes(), nw, nh, 3 * nw, QImage.Format_RGB888)
    canvas = QPixmap(size, size)
    canvas.fill(QColor(28, 28, 28))
    painter = QPainter(canvas)
    painter.drawImage((size - nw) // 2, (size - nh) // 2, qimg)
    pen = QPen(QColor(0, 210, 0) if ok else QColor(210, 0, 0), 3)
    painter.setPen(pen)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.end()
    return canvas


# ==========================================================
# バックグラウンドスレッド（選択クラスのみ検出）
# ==========================================================
class DetectWorker(QThread):
    finished = Signal(int, int, list)  # ok_count, total, rects_list

    def __init__(
        self,
        items: list,
        params: dict,
        min_area: int,
        max_area: int,
        min_circularity: float,
    ):
        super().__init__()
        self.items = items
        self.params = params
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def run(self):
        rects_list = [
            get_detections(thumb, self.params, self.min_area, self.max_area, self.min_circularity)
            for _, thumb in self.items
        ]
        ok_count = sum(1 for r in rects_list if r)
        self.finished.emit(ok_count, len(rects_list), rects_list)


# ==========================================================
# サムネイル生成バックグラウンドスレッド
# ==========================================================
class ThumbWorker(QThread):
    progress = Signal(int)
    finished = Signal(list)  # list[tuple[str, np.ndarray]]

    def __init__(self, paths: list):
        super().__init__()
        self.paths = paths

    def run(self):
        entries = []
        for i, path in enumerate(self.paths):
            img = imread_unicode(path)
            if img is not None:
                h, w = img.shape[:2]
                scale = min(THUMB_SIZE / w, THUMB_SIZE / h)
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                thumb = cv2.resize(img, (nw, nh))
                entries.append((path, thumb))
            self.progress.emit(i + 1)
        self.finished.emit(entries)


# ==========================================================
# 画像フィルタ保存スレッド
# ==========================================================
class SaveWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, int)  # saved_count, total_count

    def __init__(
        self,
        cls_name: str,
        paths: list,
        params: dict,
        thumb_min_area: int,
        thumb_max_area: int,
        min_circularity: float,
        save_mode: str,  # "copy" or "crop"
    ):
        super().__init__()
        self.cls_name = cls_name
        self.paths = paths
        self.params = params
        self.thumb_min_area = thumb_min_area
        self.thumb_max_area = thumb_max_area
        self.min_circularity = min_circularity
        self.save_mode = save_mode

    def run(self):
        dest_dir = (FILTERED_DIR if self.save_mode == "copy" else CROPPED_DIR) / self.cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        total = len(self.paths)

        for i, path in enumerate(self.paths):
            img = imread_unicode(path)
            if img is not None:
                rects = get_detections_fullsize(
                    img, self.params,
                    self.thumb_min_area, self.thumb_max_area, self.min_circularity
                )
                if rects:
                    src = Path(path)
                    if self.save_mode == "copy":
                        shutil.copy2(str(src), str(dest_dir / src.name))
                        saved += 1
                    else:
                        for j, (rx, ry, rw, rh) in enumerate(rects):
                            crop = img[ry:ry + rh, rx:rx + rw]
                            if crop.size > 0:
                                dest = dest_dir / f"{src.stem}_{j}{src.suffix}"
                                if imwrite_unicode(str(dest), crop):
                                    saved += 1
            self.progress.emit(i + 1)

        self.finished.emit(saved, total)


# ==========================================================
# サムネイルグリッド（右パネル）
# ==========================================================
class ThumbnailGrid(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(3)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self.setWidget(self._container)
        self._labels: list[QLabel] = []

    def update_grid(self, items: list, rects_list: list):
        for lbl in self._labels:
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()
        for i, ((path, thumb), rects) in enumerate(zip(items, rects_list)):
            pix = make_thumb_pixmap(thumb, THUMB_SIZE, rects)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE + 4, THUMB_SIZE + 4)
            lbl.setToolTip(os.path.basename(path))
            self._grid.addWidget(lbl, *divmod(i, GRID_COLS))
            self._labels.append(lbl)

    def clear_grid(self):
        for lbl in self._labels:
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()


# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV フィルタ保存ツール（クラスごと設定）")
        self.resize(1500, 900)

        self._current_cls: str = ""
        self._params: dict = {k: list(v) for k, v in DEFAULT_HSV.items()}
        self._min_area: int = DEFAULT_MIN_AREA
        self._max_area: int = DEFAULT_MAX_AREA
        self._min_circularity: float = DEFAULT_MIN_CIRCULARITY
        self._path_cache: dict[str, list[str]] = {}           # 起動時に収集（パスのみ）
        self._cache: dict[str, list[tuple[str, np.ndarray]]] = {}  # クラス選択時に生成
        self._rects_list: list[list] = []
        self._worker: DetectWorker | None = None
        self._thumb_worker: ThumbWorker | None = None
        self._thumb_dlg: QProgressDialog | None = None
        self._save_worker: SaveWorker | None = None
        self._save_dlg: QProgressDialog | None = None
        self._save_mode: str = ""

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_detection)

        self._build_ui()
        self._load_images()

    # --------------------------------------------------
    # UI 構築
    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ヘッダー
        header = QHBoxLayout()

        self._cls_label = QLabel("クラス: （未選択）")
        self._cls_label.setFont(QFont("", 13, QFont.Bold))
        header.addWidget(self._cls_label)

        self._detect_label = QLabel("検出率: --")
        self._detect_label.setFont(QFont("", 12))
        header.addWidget(self._detect_label)
        header.addStretch()

        btn_save_json = QPushButton("現クラスHSV設定を保存\n(json/hsv_class_{cls}.json)")
        btn_save_json.setFixedSize(210, 46)
        btn_save_json.setToolTip(
            "現在選択中のクラスの HSV パラメータを\n"
            "json/hsv_class_{クラス名}.json に保存します。"
        )
        btn_save_json.clicked.connect(self._save_class_json)
        header.addWidget(btn_save_json)

        btn_copy = QPushButton("検出画像をコピー保存\n(train用_v4_hsv_filtered/)")
        btn_copy.setFixedSize(220, 46)
        btn_copy.setToolTip(
            "HSV で検出された画像ファイルを\n"
            f"{FILTERED_DIR}/{{クラス名}}/ にコピーします。\n"
            "元ファイルはそのまま残ります。"
        )
        btn_copy.clicked.connect(lambda: self._start_save("copy"))
        header.addWidget(btn_copy)

        btn_crop = QPushButton("検出画像をクロップ保存\n(train用_v4_hsv_cropped/)")
        btn_crop.setFixedSize(220, 46)
        btn_crop.setToolTip(
            "HSV で検出された領域（バウンディングボックス）を\n"
            f"切り出して {CROPPED_DIR}/{{クラス名}}/ に保存します。"
        )
        btn_crop.clicked.connect(lambda: self._start_save("crop"))
        header.addWidget(btn_crop)

        root.addLayout(header)

        # メイン（左パネル + 右パネル）
        body = QHBoxLayout()
        body.setSpacing(6)
        root.addLayout(body)

        left = QVBoxLayout()
        left.setSpacing(6)
        body.addLayout(left, 0)
        left.addWidget(self._build_slider_group())
        left.addWidget(self._build_area_group())
        left.addWidget(self._build_class_list(), 1)

        self._grid = ThumbnailGrid()
        body.addWidget(self._grid, 1)

        self._sb = self.statusBar()
        self._sb.showMessage("画像読み込み中...")

    def _build_slider_group(self) -> QGroupBox:
        grp = QGroupBox("HSV パラメータ")
        grid = QGridLayout(grp)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        self._sliders: list[QSlider] = []
        self._val_labels: list[QLabel] = []
        for row, (name, _, _, max_val) in enumerate(SLIDER_DEFS):
            lbl_name = QLabel(name)
            lbl_name.setFixedWidth(60)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, max_val)
            s.setFixedHeight(18)
            s.valueChanged.connect(self._on_slider)
            lbl_val = QLabel("0")
            lbl_val.setFixedWidth(28)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._sliders.append(s)
            self._val_labels.append(lbl_val)
            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(s,        row, 1)
            grid.addWidget(lbl_val,  row, 2)
        return grp

    def _build_area_group(self) -> QGroupBox:
        grp = QGroupBox("形状フィルタ（サムネイル換算）")
        grid = QGridLayout(grp)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("最小面積"), 0, 0)
        self._area_slider = QSlider(Qt.Horizontal)
        self._area_slider.setRange(0, 5000)
        self._area_slider.setValue(DEFAULT_MIN_AREA)
        self._area_slider.setFixedHeight(18)
        self._area_slider.valueChanged.connect(self._on_area_min)
        self._area_label = QLabel(str(DEFAULT_MIN_AREA))
        self._area_label.setFixedWidth(48)
        self._area_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self._area_slider, 0, 1)
        grid.addWidget(self._area_label,  0, 2)

        grid.addWidget(QLabel("最大面積"), 1, 0)
        self._area_max_slider = QSlider(Qt.Horizontal)
        self._area_max_slider.setRange(0, 10000)
        self._area_max_slider.setValue(DEFAULT_MAX_AREA)
        self._area_max_slider.setFixedHeight(18)
        self._area_max_slider.valueChanged.connect(self._on_area_max)
        self._area_max_label = QLabel("無制限")
        self._area_max_label.setFixedWidth(48)
        self._area_max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self._area_max_slider, 1, 1)
        grid.addWidget(self._area_max_label,  1, 2)

        grid.addWidget(QLabel("真円度 最小"), 2, 0)
        self._circularity_slider = QSlider(Qt.Horizontal)
        self._circularity_slider.setRange(0, 100)
        self._circularity_slider.setValue(int(DEFAULT_MIN_CIRCULARITY * 100))
        self._circularity_slider.setFixedHeight(18)
        self._circularity_slider.valueChanged.connect(self._on_circularity)
        self._circularity_label = QLabel("無効")
        self._circularity_label.setFixedWidth(48)
        self._circularity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self._circularity_slider, 2, 1)
        grid.addWidget(self._circularity_label,  2, 2)

        return grp

    def _build_class_list(self) -> QGroupBox:
        grp = QGroupBox("クラス一覧（クリックで切り替え）\n✓ = HSV設定保存済み")
        v = QVBoxLayout(grp)
        self._class_list = QListWidget()
        self._class_list.setFixedWidth(270)
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        v.addWidget(self._class_list)
        return grp

    # --------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------
    def _load_images(self):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        train_path = Path(TRAIN_DIR)

        # 直下に画像がある場合はフラット構造としてフォルダ自体を1クラス扱い
        has_direct_images = any(
            f.is_file() and f.suffix.lower() in img_exts
            for f in train_path.iterdir()
        )
        if has_direct_images:
            class_dirs = [(train_path.name, train_path)]
        else:
            class_dirs = [
                (d.name, d) for d in sorted(train_path.iterdir())
                if d.is_dir() and not d.name.startswith("_")
            ]

        # ファイルパスのみ収集（サムネイル生成はクラス選択時に行う）
        for cls_name, class_dir in class_dirs:
            paths = [
                str(fpath) for fpath in sorted(class_dir.iterdir())
                if fpath.is_file() and fpath.suffix.lower() in img_exts
            ]
            if paths:
                self._path_cache[cls_name] = paths

        self._populate_class_list()
        if self._class_list.count() > 0:
            self._class_list.setCurrentRow(0)

    def _populate_class_list(self):
        self._class_list.clear()
        for cls in self._path_cache:
            marker = "✓" if has_class_config(cls) else "  "
            item = QListWidgetItem(f"{marker} {cls}")
            item.setData(Qt.UserRole, cls)
            self._class_list.addItem(item)

    # --------------------------------------------------
    # クラス選択
    # --------------------------------------------------
    def _on_class_selected(self, row: int):
        if row < 0:
            return
        item = self._class_list.item(row)
        cls = item.data(Qt.UserRole)
        if cls == self._current_cls:
            return
        self._current_cls = cls
        self._cls_label.setText(f"クラス: {cls}")

        # そのクラスの設定を読み込んでスライダーに反映
        self._params = load_class_config(cls)
        self._params_to_sliders()

        if cls in self._cache:
            # サムネイル生成済みならすぐ表示
            items = self._cache[cls]
            self._rects_list = [[] for _ in items]
            self._grid.update_grid(items, self._rects_list)
            self._run_detection()
        else:
            # 未生成ならバックグラウンドで生成
            self._grid.clear_grid()
            self._rects_list = []
            self._detect_label.setText("読み込み中...")
            paths = self._path_cache.get(cls, [])
            self._start_thumb_loading(cls, paths)

    def _start_thumb_loading(self, cls_name: str, paths: list):
        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.quit()
            self._thumb_worker.wait()
        if self._thumb_dlg:
            self._thumb_dlg.close()

        n = len(paths)
        self._sb.showMessage(f"サムネイル生成中... ({n} 枚)")
        self._thumb_dlg = QProgressDialog(
            f"画像を読み込んでいます... ({n} 枚)", None, 0, n, self
        )
        self._thumb_dlg.setWindowTitle("読み込み中")
        self._thumb_dlg.setWindowModality(Qt.WindowModal)
        self._thumb_dlg.setMinimumDuration(500)
        self._thumb_dlg.setValue(0)

        self._thumb_worker = ThumbWorker(paths)
        self._thumb_worker.progress.connect(self._thumb_dlg.setValue)
        self._thumb_worker.finished.connect(
            lambda entries: self._on_thumb_done(cls_name, entries)
        )
        self._thumb_worker.start()

    def _on_thumb_done(self, cls_name: str, entries: list):
        if self._thumb_dlg:
            self._thumb_dlg.close()
            self._thumb_dlg = None

        self._cache[cls_name] = entries

        if cls_name == self._current_cls:
            self._rects_list = [[] for _ in entries]
            self._grid.update_grid(entries, self._rects_list)
            self._run_detection()

        self._sb.showMessage(f"読み込み完了: {len(entries)} 枚")

    # --------------------------------------------------
    # スライダー操作
    # --------------------------------------------------
    def _params_to_sliders(self):
        for i, (_, key, idx, _) in enumerate(SLIDER_DEFS):
            val = self._params[key][idx]
            self._sliders[i].blockSignals(True)
            self._sliders[i].setValue(val)
            self._val_labels[i].setText(str(val))
            self._sliders[i].blockSignals(False)

    def _on_slider(self):
        vals = [s.value() for s in self._sliders]
        for i, v in enumerate(vals):
            self._val_labels[i].setText(str(v))
        h1_min, h1_max, h2_min, h2_max, s_min, s_max, v_min, v_max = vals
        self._params = {
            "lower1": [h1_min, s_min, v_min],
            "upper1": [h1_max, s_max, v_max],
            "lower2": [h2_min, s_min, v_min],
            "upper2": [h2_max, s_max, v_max],
        }
        self._debounce.start(DEBOUNCE_MS)

    def _on_area_min(self, val: int):
        self._min_area = val
        self._area_label.setText(str(val))
        self._debounce.start(DEBOUNCE_MS)

    def _on_area_max(self, val: int):
        self._max_area = val
        self._area_max_label.setText("無制限" if val == 0 else str(val))
        self._debounce.start(DEBOUNCE_MS)

    def _on_circularity(self, val: int):
        self._min_circularity = val / 100.0
        self._circularity_label.setText("無効" if val == 0 else f"{self._min_circularity:.2f}")
        self._debounce.start(DEBOUNCE_MS)

    # --------------------------------------------------
    # 検出処理（バックグラウンド）
    # --------------------------------------------------
    def _run_detection(self):
        if not self._current_cls:
            return
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        items = self._cache.get(self._current_cls, [])
        if not items:
            return
        self._sb.showMessage("処理中...")
        self._worker = DetectWorker(
            items, self._params, self._min_area, self._max_area, self._min_circularity
        )
        self._worker.finished.connect(self._on_detection_done)
        self._worker.start()

    def _on_detection_done(self, ok_count: int, total: int, rects_list: list):
        self._rects_list = rects_list
        pct = ok_count / total * 100 if total else 0
        color = "lime" if pct == 100 else "yellow" if pct >= 90 else "tomato"
        self._detect_label.setText(
            f"<span style='color:{color};font-weight:bold;'>"
            f"検出率: {pct:.1f}%  ({ok_count}/{total})"
            f"</span>"
        )
        items = self._cache.get(self._current_cls, [])
        self._grid.update_grid(items, rects_list)
        self._sb.showMessage("完了")

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    def _save_class_json(self):
        if not self._current_cls:
            QMessageBox.warning(self, "警告", "クラスを選択してください。")
            return
        save_class_config(self._current_cls, self._params)
        self._update_class_marker(self._current_cls)
        self._sb.showMessage(f"保存完了: json/hsv_class_{self._current_cls}.json")

    def _update_class_marker(self, cls_name: str):
        for i in range(self._class_list.count()):
            item = self._class_list.item(i)
            if item.data(Qt.UserRole) == cls_name:
                item.setText(f"✓ {cls_name}")
                break

    def _start_save(self, mode: str):
        if not self._current_cls:
            QMessageBox.warning(self, "警告", "クラスを選択してください。")
            return
        # SaveWorker はパスのみ使うので _path_cache から取得（サムネイル未生成でも可）
        paths = self._path_cache.get(self._current_cls, [])
        if not paths:
            QMessageBox.information(self, "情報", "このクラスに画像がありません。")
            return

        self._save_mode = mode
        mode_label = "コピー" if mode == "copy" else "クロップ"
        dest = (FILTERED_DIR if mode == "copy" else CROPPED_DIR) / self._current_cls

        reply = QMessageBox.question(
            self, "確認",
            f"クラス「{self._current_cls}」の HSV 検出画像を\n"
            f"{dest}\n"
            f"に{mode_label}保存します。よろしいですか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_dlg = QProgressDialog(
            f"画像を{mode_label}保存中...", None, 0, len(paths), self
        )
        self._save_dlg.setWindowTitle(f"{mode_label}保存中")
        self._save_dlg.setWindowModality(Qt.WindowModal)
        self._save_dlg.setMinimumDuration(0)
        self._save_dlg.setValue(0)
        self._save_dlg.show()

        self._save_worker = SaveWorker(
            self._current_cls, paths, self._params,
            self._min_area, self._max_area, self._min_circularity, mode
        )
        self._save_worker.progress.connect(self._save_dlg.setValue)
        self._save_worker.finished.connect(self._on_save_done)
        self._save_worker.start()

    def _on_save_done(self, saved: int, total: int):
        if self._save_dlg:
            self._save_dlg.close()
            self._save_dlg = None

        mode_label = "コピー" if self._save_mode == "copy" else "クロップ"
        dest = (FILTERED_DIR if self._save_mode == "copy" else CROPPED_DIR) / self._current_cls

        QMessageBox.information(
            self, "保存完了",
            f"{total} 枚を処理し、{saved} 件を{mode_label}しました。\n"
            f"保存先: {dest}"
        )
        self._sb.showMessage(f"{mode_label}保存完了: {saved}/{total} 件")


# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
