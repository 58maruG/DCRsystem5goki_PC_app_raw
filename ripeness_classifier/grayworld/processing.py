"""画像処理（グレーワールド正規化・未熟/過熟占有率の算出とサムネイル描画）。

判定に使う特徴量の算出（compute_ripeness）・分類（classify_3）は本体(base)のロジックを
そのまま再利用する。grayworld 版に固有なのは apply_grayworld による色温度補正の前処理のみ。
"""

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont

from ripeness_classifier.common.mask_utils import (
    mask_from_hsv, remove_stem, remove_reflection_sat,
)
from ripeness_classifier.base.processing import compute_ripeness
from ripeness_classifier.grayworld.classify import classify_3
from ripeness_classifier.grayworld.config import (
    COLOR_HEALTHY, COLOR_UNRIPE, COLOR_OVERRIPE, COLOR_NONE,
)

__all__ = ["apply_grayworld", "compute_ripeness", "make_thumb_pixmap"]


def apply_grayworld(img: np.ndarray) -> np.ndarray:
    """per-image グレーワールド正規化（BGR）。
    各チャンネル平均を全体グレー平均に揃えるスケールを掛け、白色点を中性化する。
    カメラ間の色温度（ホワイトバランス）差を吸収し、同じ果実のHSVを近づけるのが狙い。
    再着色のみで座標は動かさないため、マスク座標は元画像にそのまま流用できる。
    注意: 画像全体の平均を使うので、背景が多色だと補正が偏る（既知の限界）。"""
    b, g, r = cv2.split(img.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m_gray = (mb + mg + mr) / 3.0
    eps = 1e-6
    b *= m_gray / (mb + eps)
    g *= m_gray / (mg + eps)
    r *= m_gray / (mr + eps)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def make_thumb_pixmap(
    thumb: np.ndarray,
    size: int,
    occ_unripe: float,
    occ_overripe: float,
    fruit_found: bool,
    rects: list,
    occ_unripe_confirm: float,
    occ_overripe_confirm: float,
    debug: bool = False,
    fruit_params: dict | None = None,
    unripe_params: dict | None = None,
    overripe_params: dict | None = None,
    stem_open: int = 0,
    reflect_sat_lo: int = 0,
    reflect_sat_hi: int = 255,
    reflect_v_lo: int = 0,
    reflect_v_hi: int = 255,
    grayworld: bool = False,
) -> QPixmap:
    """3クラス判定で枠色を決定し、左上に未熟占有率・過熟占有率を描画したサムネイルを返す。
    grayworld=True のとき、表示・マスク描画を正規化後の色で行う（判定と一致させるため）。
    debug=True のとき、マスクの各段階の輪郭を色分けで重ね描く（どの段で削れたか可視化）:
      白=生マスク / 黄=虚像除去後 / 桃=果柄除去後(最終) / 緑=クロップ枠 /
      青=未熟マスク(果実内) / 黒=過熟マスク(果実内)。"""
    cls = classify_3(occ_unripe, occ_overripe, fruit_found, occ_unripe_confirm, occ_overripe_confirm)
    if cls == "none":
        border_color = COLOR_NONE
        label_text   = "---"
    else:
        border_color = {
            "unripe":   COLOR_UNRIPE,
            "overripe": COLOR_OVERRIPE,
            "healthy":  COLOR_HEALTHY,
        }[cls]
        # 未熟占有率% / 過熟占有率% を表示（閾値調整の手がかり）
        label_text = f"未熟{occ_unripe * 100:.0f}% 過熟{occ_overripe * 100:.0f}%"

    # 判定と表示を揃えるため、正規化後の色を表示に使う（座標は不変）
    disp = apply_grayworld(thumb) if grayworld else thumb
    annotated = disp.copy()
    for (rx, ry, rw, rh) in rects:
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 0), 1)

    # デバッグ表示: マスクの3段階（生→虚像除去→果柄除去）＋未熟/過熟マスクの輪郭を色分けで重ねる
    if debug and fruit_params is not None:
        hsv_t  = cv2.cvtColor(disp, cv2.COLOR_BGR2HSV)
        m_raw  = mask_from_hsv(hsv_t, fruit_params)                    # 生マスク（開閉処理まで）
        m_ref  = remove_reflection_sat(
            m_raw, hsv_t[:, :, 1], hsv_t[:, :, 2],
            reflect_sat_lo, reflect_sat_hi, reflect_v_lo, reflect_v_hi)  # 虚像除去後
        m_stem = remove_stem(m_ref, stem_open)                        # 果柄除去後（最終）
        stages = [
            (m_raw,  (255, 255, 255)),   # 白: 生
            (m_ref,  (0, 255, 255)),     # 黄: 虚像除去後
            (m_stem, (255, 0, 255)),     # 桃: 果柄除去後（最終）
        ]
        # 未熟マスク（果実本体内に限定）＝未熟の色相・占有率の算出対象。青で重ねる。
        if unripe_params is not None:
            m_unripe = cv2.bitwise_and(mask_from_hsv(hsv_t, unripe_params), m_stem)
            stages.append((m_unripe, (255, 0, 0)))  # 青(BGR): 未熟マスク（果実内）
        # 過熟マスク（果実本体内に限定）＝過熟の色相・明度・占有率の算出対象。黒で重ねる。
        if overripe_params is not None:
            m_overripe = cv2.bitwise_and(mask_from_hsv(hsv_t, overripe_params), m_stem)
            stages.append((m_overripe, (0, 0, 0)))  # 黒: 過熟マスク（果実内）
        for m, col in stages:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, cnts, -1, col, 1)

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
    pen = QPen(border_color, 3)
    painter.setPen(pen)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.setFont(QFont("", 9, QFont.Bold))
    painter.setPen(QPen(border_color))
    painter.drawText(4, 13, label_text)
    painter.end()
    return canvas
