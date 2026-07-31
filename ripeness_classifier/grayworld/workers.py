"""バックグラウンドスレッド（サムネイル生成・スコア計算・保存。グレーワールド版）。

特徴量算出・判定は本体(base)のロジックをそのまま使う。grayworld 版に固有なのは
apply_grayworld による色温度補正を判定前に挟むかどうかのみ。
"""

import shutil
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from ripeness_classifier.common.io_utils import imread_unicode, imwrite_unicode
from ripeness_classifier.grayworld.config import THUMB_SIZE, _DEST_ROOTS
from ripeness_classifier.grayworld.processing import apply_grayworld, compute_ripeness
from ripeness_classifier.grayworld.classify import classify_3


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
                ih, iw = img.shape[:2]
                scale = min(THUMB_SIZE / iw, THUMB_SIZE / ih)
                nw = max(1, int(iw * scale))
                nh = max(1, int(ih * scale))
                thumb = cv2.resize(img, (nw, nh))
                entries.append((path, thumb))
            self.progress.emit(i + 1)
        self.finished.emit(entries)


# ==========================================================
# スコア計算スレッド（選択クラスのみ）
# ==========================================================
class ScoreWorker(QThread):
    # occ_unripes, occ_overripes, fruit_founds, rects_list
    finished = Signal(list, list, list, list)

    def __init__(
        self,
        items: list,
        fruit_params: dict,
        unripe_params: dict,
        overripe_params: dict,
        min_area: int,
        stem_open: int,
        reflect_sat_lo: int,
        reflect_sat_hi: int,
        reflect_v_lo: int,
        reflect_v_hi: int,
        grayworld: bool,
    ):
        super().__init__()
        self.items = items
        self.fruit_params = fruit_params
        self.unripe_params = unripe_params
        self.overripe_params = overripe_params
        self.min_area = min_area
        self.stem_open = stem_open
        self.reflect_sat_lo = reflect_sat_lo
        self.reflect_sat_hi = reflect_sat_hi
        self.reflect_v_lo = reflect_v_lo
        self.reflect_v_hi = reflect_v_hi
        self.grayworld = grayworld

    def run(self):
        occ_unripes, occ_overripes, fruit_founds, rects_list = [], [], [], []
        for _, thumb in self.items:
            frame = apply_grayworld(thumb) if self.grayworld else thumb
            ou, oo, ff, r = compute_ripeness(
                frame, self.fruit_params, self.unripe_params, self.overripe_params,
                self.min_area, self.stem_open,
                self.reflect_sat_lo, self.reflect_sat_hi, self.reflect_v_lo, self.reflect_v_hi
            )
            occ_unripes.append(ou)
            occ_overripes.append(oo)
            fruit_founds.append(ff)
            rects_list.append(r)
        self.finished.emit(occ_unripes, occ_overripes, fruit_founds, rects_list)


# ==========================================================
# 保存スレッド（プレビューと同一のサムネイル基準で判定し、クロップは元画像から）
# ==========================================================
class SaveWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, int)  # saved_count, total

    def __init__(
        self,
        cls_name: str,
        paths: list,
        fruit_params: dict,
        unripe_params: dict,
        overripe_params: dict,
        min_area: int,
        stem_open: int,          # 果柄除去（開処理）半径（サムネイル換算px）
        reflect_sat_lo: int,     # 虚像除去の彩度範囲下限（0〜255, スケール非依存）
        reflect_sat_hi: int,     # 虚像除去の彩度範囲上限（0〜255, スケール非依存）
        reflect_v_lo: int,       # 虚像除去の明度範囲下限（0〜255, 試験導入）
        reflect_v_hi: int,       # 虚像除去の明度範囲上限（0〜255, 試験導入）
        grayworld: bool,      # グレーワールド正規化を判定に適用するか
        occ_unripe_confirm: float,    # 未熟確定の占有率境界 0.0〜1.0
        occ_overripe_confirm: float,  # 過熟確定の占有率境界 0.0〜1.0
        target: str,          # "healthy" | "unripe" | "overripe"
        save_mode: str,       # "copy" or "crop"
    ):
        super().__init__()
        self.cls_name = cls_name
        self.paths = paths
        self.fruit_params = fruit_params
        self.unripe_params = unripe_params
        self.overripe_params = overripe_params
        self.min_area = min_area
        self.stem_open = stem_open
        self.reflect_sat_lo = reflect_sat_lo
        self.reflect_sat_hi = reflect_sat_hi
        self.reflect_v_lo = reflect_v_lo
        self.reflect_v_hi = reflect_v_hi
        self.grayworld = grayworld
        self.occ_unripe_confirm = occ_unripe_confirm
        self.occ_overripe_confirm = occ_overripe_confirm
        self.target = target
        self.save_mode = save_mode

    def run(self):
        dest_dir = _DEST_ROOTS[self.target] / self.cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = 0

        for i, path in enumerate(self.paths):
            img = imread_unicode(path)
            if img is not None:
                ih, iw = img.shape[:2]
                # プレビュー(ScoreWorker)と同一のサムネイルを生成し、同じ基準で判定する。
                #   これで左上の個数表示と保存枚数が必ず一致する（判定は112px基準で統一）。
                scale = min(THUMB_SIZE / iw, THUMB_SIZE / ih)
                nw = max(1, int(iw * scale))
                nh = max(1, int(ih * scale))
                thumb = cv2.resize(img, (nw, nh))
                # 判定はプレビューと同じく正規化後のサムネイルで行う（クロップは元画像から）
                frame = apply_grayworld(thumb) if self.grayworld else thumb

                occ_unripe, occ_overripe, fruit_found, rects = compute_ripeness(
                    frame, self.fruit_params, self.unripe_params, self.overripe_params,
                    self.min_area, self.stem_open,
                    self.reflect_sat_lo, self.reflect_sat_hi, self.reflect_v_lo, self.reflect_v_hi
                )

                cls = classify_3(occ_unripe, occ_overripe, fruit_found,
                                 self.occ_unripe_confirm, self.occ_overripe_confirm)
                if cls == self.target:
                    src = Path(path)
                    if self.save_mode == "copy":
                        shutil.copy2(str(src), str(dest_dir / src.name))
                        saved += 1
                    else:
                        # サムネイル基準の矩形を元画像スケールへ拡大してクロップする
                        for j, (rx, ry, rw, rh) in enumerate(rects):
                            fx0 = max(0,  int(rx / scale))
                            fy0 = max(0,  int(ry / scale))
                            fx1 = min(iw, int((rx + rw) / scale))
                            fy1 = min(ih, int((ry + rh) / scale))
                            crop = img[fy0:fy1, fx0:fx1]
                            if crop.size > 0:
                                dest = dest_dir / f"{src.stem}_{j}{src.suffix}"
                                if imwrite_unicode(str(dest), crop):
                                    saved += 1

            self.progress.emit(i + 1)

        self.finished.emit(saved, len(self.paths))
