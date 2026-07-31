"""再利用可能ウィジェット（HSVレンジスライダー・サムネイルグリッド・伸縮ヘルパー）。"""

import os

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QSpinBox, QGridLayout, QGroupBox,
    QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QObject, QEvent, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen

from ripeness_classifier.base.config import THUMB_SIZE, GRID_COLS
from ripeness_classifier.base.processing import make_thumb_pixmap


# ==========================================================
# スライダー行ヘルパー（キーボード数値入力対応）
# ==========================================================
def make_slider_row(minimum: int, maximum: int, value: int,
                    suffix: str = "", special_zero: bool = False) -> tuple[QSlider, QSpinBox]:
    """QSlider と QSpinBox を双方向同期して返す。スピンボックスにキーボードで直接数値を入力できる。
      suffix       : スピンボックスに付ける単位表示（例 "%"）
      special_zero : True で最小値(0)のとき "OFF" と表示する
    QSlider/QSpinBox は同値への setValue では valueChanged を再発火しないため、
    互いの valueChanged→setValue を相互接続してもループしない（Qtの標準的な同期パターン）。"""
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    slider.setFixedHeight(18)

    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setFixedWidth(56)
    spin.setButtonSymbols(QSpinBox.NoButtons)   # 上下ボタンを消して省スペース化（キー入力・ホイールは有効）
    spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    # 複数桁を入力する途中（例: "150"の"1"だけ打った瞬間）でスライダーが飛ばないよう、
    #   確定（Enter/フォーカスアウト）まで valueChanged を発火しない。矢印キー・スピンボタンは影響を受けない。
    spin.setKeyboardTracking(False)
    if suffix:
        spin.setSuffix(suffix)
    if special_zero:
        spin.setSpecialValueText("OFF")

    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    return slider, spin


# ==========================================================
# 範囲スライダー（左右のハンドルで下限・上限を指定する1本のバー）
# ==========================================================
class RangeSlider(QWidget):
    """左右2つのハンドルをドラッグして下限・上限を同時に指定するスライダー。
    最小値側ハンドルと最大値側ハンドルの位置に、クリック/ドラッグ位置に近い方が反応する。
    キーボード操作: フォーカスを当てて ←→/↑↓ で「最後に操作した側のハンドル」を1ずつ、
    Shift+←→/↑↓ で10ずつ動かせる（QSlider標準のキー操作に倣った挙動）。
    フォーカス中はキー操作の対象ハンドルを枠線でハイライトする。"""
    rangeChanged = Signal(int, int)  # (low, high)

    _HANDLE_R = 7
    _KEY_STEP = 1
    _KEY_STEP_SHIFT = 10

    def __init__(self, minimum: int, maximum: int, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._low = minimum
        self._high = maximum
        self._active = None  # "low" | "high" | None（ドラッグ中のみ）
        self._kbd_target = "low"  # "low" | "high"（キー操作の対象。クリックでも更新され、ドラッグ終了後も保持）
        self.setFixedHeight(18)
        self.setMinimumWidth(60)
        self.setFocusPolicy(Qt.StrongFocus)  # Tabフォーカス・クリックフォーカスの両方でキー操作可能にする

    def low(self) -> int:
        return self._low

    def high(self) -> int:
        return self._high

    def setRange(self, low: int, high: int):
        low = max(self._min, min(int(low), self._max))
        high = max(self._min, min(int(high), self._max))
        if low > high:
            low, high = high, low
        changed = (low != self._low) or (high != self._high)
        self._low, self._high = low, high
        self.update()
        if changed:
            self.rangeChanged.emit(self._low, self._high)

    def _usable_width(self) -> int:
        return max(1, self.width() - 2 * self._HANDLE_R)

    def _value_to_x(self, value: int) -> int:
        span = self._max - self._min
        if span <= 0:
            return self._HANDLE_R
        return self._HANDLE_R + round((value - self._min) / span * self._usable_width())

    def _x_to_value(self, x: float) -> int:
        span = self._max - self._min
        ratio = (x - self._HANDLE_R) / self._usable_width()
        ratio = max(0.0, min(1.0, ratio))
        return round(self._min + ratio * span)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        mid_y = self.height() // 2
        painter.setPen(Qt.NoPen)

        # 溝（全体）
        painter.setBrush(QColor(110, 110, 110))
        painter.drawRoundedRect(QRect(self._HANDLE_R, mid_y - 2, self._usable_width(), 4), 2, 2)

        # 選択範囲（下限〜上限）
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        painter.setBrush(QColor(90, 160, 250))
        painter.drawRoundedRect(QRect(x_low, mid_y - 2, max(1, x_high - x_low), 4), 2, 2)

        # ハンドル（フォーカス中はキー操作の対象ハンドルを太い青枠でハイライト）
        has_focus = self.hasFocus()
        for side, x in (("low", x_low), ("high", x_high)):
            painter.setBrush(QColor(235, 235, 235))
            if has_focus and side == self._kbd_target:
                painter.setPen(QPen(QColor(60, 140, 255), 2))
            else:
                painter.setPen(QColor(70, 70, 70))
            painter.drawEllipse(QPoint(x, mid_y), self._HANDLE_R, self._HANDLE_R)
            painter.setPen(Qt.NoPen)
        painter.end()

    def _event_x(self, event) -> float:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return pos.x()

    def mousePressEvent(self, event):
        self.setFocus(Qt.MouseFocusReason)
        x = self._event_x(event)
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        self._active = "low" if abs(x - x_low) <= abs(x - x_high) else "high"
        self._kbd_target = self._active  # クリックした側をキー操作の対象にも合わせる
        self._drag_to(x)

    def mouseMoveEvent(self, event):
        if self._active is None:
            return
        self._drag_to(self._event_x(event))

    def mouseReleaseEvent(self, event):
        self._active = None

    def _drag_to(self, x: float):
        val = self._x_to_value(x)
        if self._active == "low":
            self.setRange(min(val, self._high), self._high)
        else:
            self.setRange(self._low, max(val, self._low))

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update()  # ハンドルのフォーカスハイライトを表示

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update()  # フォーカスハイライトを消す

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Down):
            delta = -1
        elif key in (Qt.Key_Right, Qt.Key_Up):
            delta = 1
        else:
            super().keyPressEvent(event)
            return
        step = self._KEY_STEP_SHIFT if (event.modifiers() & Qt.ShiftModifier) else self._KEY_STEP
        if self._kbd_target == "low":
            self.setRange(self._low + delta * step, self._high)
        else:
            self.setRange(self._low, self._high + delta * step)
        event.accept()


def _make_range_spin(minimum: int, maximum: int) -> QSpinBox:
    """RangeSlider用の下限/上限スピンボックスを1個作る（内部ヘルパー）。
    キーボード入力中は1文字ごとに valueChanged を発火しない（Enter/フォーカスアウトで確定）。
      有効のままだと例えば上限欄に "132" と打つ途中で一瞬 "1" になった時点で
      valueChanged(1) が飛び、下限より小さい値になって RangeSlider.setRange() の
      low/high 入れ替えが誤発動し、下限側が書き換わったように見えるバグになる。
      矢印キー・スピンボタンでの増減は無効化の影響を受けず即時に反映される。"""
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setFixedWidth(46)
    spin.setButtonSymbols(QSpinBox.NoButtons)
    spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    spin.setKeyboardTracking(False)
    return spin


def make_range_slider_row(minimum: int, maximum: int, low: int, high: int) -> tuple[RangeSlider, QSpinBox, QSpinBox]:
    """RangeSlider（左右ハンドルで下限・上限を指定）と、それに双方向同期する下限/上限スピンボックスを返す。
    HsvSliderGroup内の各チャンネル行と同じ構成を、グループに属さない単独の行としても使えるよう
    切り出したもの（検出フィルタの虚像除去S/Vなど、HSVマスクではない範囲パラメータ向け）。"""
    rs = RangeSlider(minimum, maximum)
    lo_spin = _make_range_spin(minimum, maximum)
    hi_spin = _make_range_spin(minimum, maximum)
    rs.setRange(low, high)
    lo_spin.setValue(rs.low())
    hi_spin.setValue(rs.high())

    def _sync_spins(_lo: int = 0, _hi: int = 0):
        lo_spin.blockSignals(True)
        hi_spin.blockSignals(True)
        lo_spin.setValue(rs.low())
        hi_spin.setValue(rs.high())
        lo_spin.blockSignals(False)
        hi_spin.blockSignals(False)

    rs.rangeChanged.connect(_sync_spins)
    lo_spin.valueChanged.connect(lambda v: rs.setRange(v, rs.high()))
    hi_spin.valueChanged.connect(lambda v: rs.setRange(rs.low(), v))
    return rs, lo_spin, hi_spin


# ==========================================================
# HSVスライダーグループ（再利用可能ウィジェット）
# ==========================================================
class HsvSliderGroup(QGroupBox):
    changed = Signal()

    # (表示名, 最小, 最大)。H1/H2は赤の折り返し（低H側・高H側）に対応する2本、S/Vは1本ずつ。
    _CHANNEL_DEFS_DUAL_HUE = [
        ("H1", 0, 180),
        ("H2", 0, 180),
        ("S",  0, 255),
        ("V",  0, 255),
    ]
    # 折り返しが不要な色相（黄〜緑など）用: Hは1本だけ表示する。
    _CHANNEL_DEFS_SINGLE_HUE = [
        ("H",  0, 180),
        ("S",  0, 255),
        ("V",  0, 255),
    ]

    def __init__(self, title: str, default: dict, parent=None, single_hue: bool = False):
        """single_hue=True で色相バーを1本だけにする（0/180をまたがない色向け）。
        内部的には get_params() が lower2/upper2 に lower1/upper1 と同じ値を複製するので、
        mask_from_hsv 側の2帯構成はそのまま変えずに済む。"""
        super().__init__(title, parent)
        self._single_hue = single_hue
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)      # 行どうしの縦の隙間を最小限に
        grid.setContentsMargins(6, 2, 6, 2)
        grid.setColumnStretch(1, 1)
        self._range_sliders: dict[str, RangeSlider] = {}
        self._low_spins: dict[str, QSpinBox] = {}
        self._high_spins: dict[str, QSpinBox] = {}

        channel_defs = self._CHANNEL_DEFS_SINGLE_HUE if single_hue else self._CHANNEL_DEFS_DUAL_HUE
        for row, (name, lo, hi) in enumerate(channel_defs):
            lbl = QLabel(name)
            lbl.setFixedWidth(20)
            rs = RangeSlider(lo, hi)
            # 下限・上限それぞれをキーボードから直接入力できるスピンボックス
            lo_spin = self._make_spin(lo, hi)
            hi_spin = self._make_spin(lo, hi)
            rs.rangeChanged.connect(lambda _lo, _hi, n=name: self._on_range_changed(n))
            lo_spin.valueChanged.connect(lambda v, n=name: self._on_low_spin(n, v))
            hi_spin.valueChanged.connect(lambda v, n=name: self._on_high_spin(n, v))
            self._range_sliders[name] = rs
            self._low_spins[name] = lo_spin
            self._high_spins[name] = hi_spin
            grid.addWidget(lbl, row, 0)
            grid.addWidget(rs,  row, 1)
            grid.addWidget(lo_spin, row, 2)
            grid.addWidget(hi_spin, row, 3)

        self.set_params(default)

    @staticmethod
    def _make_spin(minimum: int, maximum: int) -> QSpinBox:
        return _make_range_spin(minimum, maximum)

    def _sync_spins(self, name: str):
        """スライダーの現在値をスピンボックスへ反映（信号を止めてループを防ぐ）。"""
        rs = self._range_sliders[name]
        lo_spin, hi_spin = self._low_spins[name], self._high_spins[name]
        for spin, val in ((lo_spin, rs.low()), (hi_spin, rs.high())):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def _on_range_changed(self, name: str):
        # スライダー（ドラッグ）変更 → スピン表示を更新して changed を通知
        self._sync_spins(name)
        self.changed.emit()

    def _on_low_spin(self, name: str, val: int):
        # スピン入力 → スライダーへ反映（rangeChanged 経由で表示同期・changed 通知される）
        rs = self._range_sliders[name]
        rs.setRange(val, rs.high())

    def _on_high_spin(self, name: str, val: int):
        rs = self._range_sliders[name]
        rs.setRange(rs.low(), val)

    def set_params(self, params: dict):
        if self._single_hue:
            self._range_sliders["H"].setRange(params["lower1"][0], params["upper1"][0])
        else:
            self._range_sliders["H1"].setRange(params["lower1"][0], params["upper1"][0])
            self._range_sliders["H2"].setRange(params["lower2"][0], params["upper2"][0])
        self._range_sliders["S"].setRange(params["lower1"][1], params["upper1"][1])
        self._range_sliders["V"].setRange(params["lower1"][2], params["upper1"][2])
        for name in self._range_sliders:
            self._sync_spins(name)

    def get_params(self) -> dict:
        s, v = self._range_sliders["S"], self._range_sliders["V"]
        if self._single_hue:
            h = self._range_sliders["H"]
            return {
                "lower1": [h.low(),  s.low(), v.low()],
                "upper1": [h.high(), s.high(), v.high()],
                "lower2": [h.low(),  s.low(), v.low()],
                "upper2": [h.high(), s.high(), v.high()],
            }
        h1, h2 = self._range_sliders["H1"], self._range_sliders["H2"]
        return {
            "lower1": [h1.low(),  s.low(), v.low()],
            "upper1": [h1.high(), s.high(), v.high()],
            "lower2": [h2.low(),  s.low(), v.low()],
            "upper2": [h2.high(), s.high(), v.high()],
        }


# ==========================================================
# サムネイルグリッド
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
        # デバッグ描画用コンテキスト（update_grid 前に set_debug_context で更新）
        self._debug: bool = False
        self._dbg_fruit_params: dict | None = None
        self._dbg_unripe_params: dict | None = None
        self._dbg_overripe_params: dict | None = None
        self._dbg_stem_open: int = 0
        self._dbg_reflect_sat_lo: int = 0
        self._dbg_reflect_sat_hi: int = 255
        self._dbg_reflect_v_lo: int = 0
        self._dbg_reflect_v_hi: int = 255

    def set_debug_context(self, debug: bool, fruit_params: dict, unripe_params: dict, overripe_params: dict,
                          stem_open: int,
                          reflect_sat_lo: int = 0, reflect_sat_hi: int = 255,
                          reflect_v_lo: int = 0, reflect_v_hi: int = 255):
        self._debug = debug
        self._dbg_fruit_params = fruit_params
        self._dbg_unripe_params = unripe_params
        self._dbg_overripe_params = overripe_params
        self._dbg_stem_open = stem_open
        self._dbg_reflect_sat_lo = reflect_sat_lo
        self._dbg_reflect_sat_hi = reflect_sat_hi
        self._dbg_reflect_v_lo = reflect_v_lo
        self._dbg_reflect_v_hi = reflect_v_hi

    def update_grid(
        self,
        items: list,
        occ_unripes: list,
        occ_overripes: list,
        fruit_founds: list,
        rects_list: list,
        occ_unripe_confirm: float,
        occ_overripe_confirm: float,
    ):
        # ウィジェットの破棄・再生成を避け、既存ラベルの中身だけ差し替える。
        #   これで枚数が変わらない限りレイアウトは不変となり、スクロール位置が維持される
        #   （閾値境界など枠色だけの変更でスクロールが先頭へ戻る問題の対策）。
        n = len(items)
        while len(self._labels) > n:                     # 余ったラベルは末尾から削除
            lbl = self._labels.pop()
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        while len(self._labels) < n:                     # 足りない分だけ追加
            lbl = QLabel()
            lbl.setFixedSize(THUMB_SIZE + 4, THUMB_SIZE + 4)
            self._grid.addWidget(lbl, *divmod(len(self._labels), GRID_COLS))
            self._labels.append(lbl)

        for i, ((path, thumb), ou, oo, ff, rects) in enumerate(
            zip(items, occ_unripes, occ_overripes, fruit_founds, rects_list)
        ):
            pix = make_thumb_pixmap(
                thumb, THUMB_SIZE, ou, oo, ff, rects,
                occ_unripe_confirm, occ_overripe_confirm,
                self._debug, self._dbg_fruit_params, self._dbg_unripe_params, self._dbg_overripe_params,
                self._dbg_stem_open,
                self._dbg_reflect_sat_lo, self._dbg_reflect_sat_hi,
                self._dbg_reflect_v_lo, self._dbg_reflect_v_hi
            )
            lbl = self._labels[i]
            lbl.setPixmap(pix)
            lbl.setToolTip(
                f"{os.path.basename(path)}\n"
                f"未熟占有率: {ou * 100:.1f}%  /  過熟占有率: {oo * 100:.1f}%"
            )

    def clear_grid(self):
        for lbl in self._labels:
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()


# ==========================================================
# 伸縮（ドラッグ・リサイズ）対応ヘルパー
# ==========================================================
def scroll_wrap(w: QWidget, min_height: int = 30) -> QScrollArea:
    """ウィジェットをスクロール領域で包んで返す。
    スプリッタでブロックを内容より小さく縮めても、縦スクロールバーで全内容に届く。
    横はビューポート幅に追従（widgetResizable）させ、横スクロールは出さない。"""
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(w)
    sa.setMinimumHeight(min_height)
    return sa


class BlockZoomFilter(QObject):
    """Ctrl+ホイールで対象ブロックの文字サイズ・列幅・最小幅をズームするイベントフィルタ。
    ブロックと全子ウィジェットに自身を仕掛け、Ctrl+ホイールだけを捕捉して拡縮する
    （Ctrl なしのホイールはスライダー操作・スクロールとしてそのまま通す）。
      block        : ズーム対象のグループ（フォントは子へ継承される）
      width_target : 幅の下限を広げる対象（スクロール領域＝左パネルの幅に波及）
    setFixedWidth/Height は min==max になる性質を使い、基準値を記憶して比率で拡縮する。"""

    def __init__(self, block: QWidget, width_target: QWidget):
        super().__init__(block)
        self._block = block
        self._wtarget = width_target
        self._scale = 1.0
        pt = block.font().pointSizeF()
        self._base_pt = pt if pt > 0 else float(QApplication.font().pointSize() or 9)
        self._fixed_w = [
            (w, w.minimumWidth()) for w in block.findChildren(QLabel)
            if w.minimumWidth() == w.maximumWidth() and w.minimumWidth() > 0
        ]
        # QSlider・RangeSlider どちらも高さ固定ウィジェットとしてズーム対象にする
        height_fixed_widgets = list(block.findChildren(QSlider)) + list(block.findChildren(RangeSlider))
        self._fixed_h = [
            (s, s.minimumHeight()) for s in height_fixed_widgets
            if s.minimumHeight() == s.maximumHeight() and s.minimumHeight() > 0
        ]
        self._base_w = max(block.sizeHint().width(), 200)

        block.installEventFilter(self)
        for w in block.findChildren(QWidget):
            w.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and (ev.modifiers() & Qt.ControlModifier):
            dy = ev.angleDelta().y()
            if dy != 0:
                factor = 1.1 if dy > 0 else 1.0 / 1.1
                self._scale = max(0.6, min(2.5, self._scale * factor))
                self._apply()
            return True   # Ctrl+ホイールは消費（スライダー値変更・スクロールを防ぐ）
        return False

    def _apply(self):
        f = self._block.font()
        f.setPointSizeF(self._base_pt * self._scale)
        self._block.setFont(f)                       # 子ラベル・スライダーへ継承
        for w, bw in self._fixed_w:
            w.setFixedWidth(max(1, round(bw * self._scale)))
        for s, bh in self._fixed_h:
            s.setFixedHeight(max(8, round(bh * self._scale)))
        self._wtarget.setMinimumWidth(round(self._base_w * self._scale))
