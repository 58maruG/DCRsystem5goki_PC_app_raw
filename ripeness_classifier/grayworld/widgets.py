"""再利用可能ウィジェット（グレーワールド版）。

HSVスライダーは本体(base)のものをそのまま再利用する。サムネイルグリッドは
グレーワールド正規化の表示切り替えを扱うため、base.widgets.ThumbnailGrid を継承して
描画部分だけ差し替える。
"""

import os

from PySide6.QtWidgets import QLabel

from ripeness_classifier.base.widgets import HsvSliderGroup, ThumbnailGrid as _BaseThumbnailGrid
from ripeness_classifier.grayworld.config import THUMB_SIZE, GRID_COLS
from ripeness_classifier.grayworld.processing import make_thumb_pixmap

__all__ = ["HsvSliderGroup", "ThumbnailGrid"]


# ==========================================================
# サムネイルグリッド（グレーワールド対応）
# ==========================================================
class ThumbnailGrid(_BaseThumbnailGrid):
    def __init__(self):
        super().__init__()
        self._dbg_grayworld: bool = False

    def set_debug_context(self, debug: bool, fruit_params: dict, unripe_params: dict, overripe_params: dict,
                          stem_open: int,
                          reflect_sat_lo: int = 0, reflect_sat_hi: int = 255,
                          reflect_v_lo: int = 0, reflect_v_hi: int = 255,
                          grayworld: bool = False):
        super().set_debug_context(
            debug, fruit_params, unripe_params, overripe_params, stem_open,
            reflect_sat_lo, reflect_sat_hi, reflect_v_lo, reflect_v_hi)
        self._dbg_grayworld = grayworld

    def update_grid(
        self,
        items: list,
        occ_unripes: list,
        occ_overripes: list,
        fruit_founds: list,
        rects_list: list,
        occ_unripe_confirm: float,
        occ_overripe_confirm: float,
        grayworld: bool = False,
    ):
        # ウィジェットを破棄せず既存ラベルの中身だけ差し替える（スクロール位置を維持）
        n = len(items)
        while len(self._labels) > n:
            lbl = self._labels.pop()
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        while len(self._labels) < n:
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
                self._dbg_reflect_v_lo, self._dbg_reflect_v_hi,
                grayworld,
            )
            lbl = self._labels[i]
            lbl.setPixmap(pix)
            lbl.setToolTip(
                f"{os.path.basename(path)}\n"
                f"未熟占有率: {ou * 100:.1f}%  /  過熟占有率: {oo * 100:.1f}%"
            )
