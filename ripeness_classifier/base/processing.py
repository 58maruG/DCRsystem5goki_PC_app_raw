"""画像処理（未熟占有率・過熟占有率の算出とサムネイル描画）。"""

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont

from ripeness_classifier.common.mask_utils import (
    mask_from_hsv, remove_stem, remove_reflection_sat,
)
from ripeness_classifier.base.classify import classify_3
from ripeness_classifier.base.config import COLOR_HEALTHY, COLOR_UNRIPE, COLOR_OVERRIPE, COLOR_NONE


def compute_ripeness(
    frame: np.ndarray,
    fruit_params: dict,
    unripe_params: dict,
    overripe_params: dict,
    min_area: int,
    stem_open: int = 0,
    reflect_sat_lo: int = 0,
    reflect_sat_hi: int = 255,
    reflect_v_lo: int = 0,
    reflect_v_hi: int = 255,
) -> tuple[float, float, bool, list]:
    """
    Returns:
        occ_unripe   : 未熟占有率 (0.0〜1.0)。未熟マスク∩果実マスクの面積 ÷ 果実マスク面積
        occ_overripe : 過熟占有率 (0.0〜1.0)。過熟マスク∩果実マスクの面積 ÷ 果実マスク面積
        fruit_found  : min_area 以上の果実領域が存在するか
        rects        : 果実矩形リスト（統合後の1矩形。該当なしなら空）
    stem_open               : 果柄除去の開処理半径（px）。細い果柄を切り離して果実本体だけ残す。
    reflect_sat_lo/hi       : 虚像除去の彩度範囲（0〜255）。淡い反射(低彩度)を落として実体だけ残す。
    reflect_v_lo/hi         : 虚像除去の明度範囲（0〜255・試験導入）。明るい反射(高明度)を落とす。
    """
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fruit_mask = mask_from_hsv(hsv, fruit_params)
    # アクリル板の虚像（低彩度・高明度の白飛び）を彩度範囲／明度範囲で除去
    #   → 縦双子の房境界は高彩度なので保たれる
    fruit_mask = remove_reflection_sat(
        fruit_mask, hsv[:, :, 1], hsv[:, :, 2],
        reflect_sat_lo, reflect_sat_hi, reflect_v_lo, reflect_v_hi)
    # 果柄（細い突起）を開処理で除去 → 以降の占有率・輪郭・クロップは全て本体基準になる
    fruit_mask = remove_stem(fruit_mask, stem_open)

    unripe_mask   = mask_from_hsv(hsv, unripe_params)
    overripe_mask = mask_from_hsv(hsv, overripe_params)
    # 果実マスク内に限定（＝果実の中の未熟色／過熟色だけを数える）
    unripe_in_fruit   = cv2.bitwise_and(unripe_mask, fruit_mask)
    overripe_in_fruit = cv2.bitwise_and(overripe_mask, fruit_mask)

    fruit_area    = int(np.count_nonzero(fruit_mask))
    unripe_area   = int(np.count_nonzero(unripe_in_fruit))
    overripe_area = int(np.count_nonzero(overripe_in_fruit))
    occ_unripe   = unripe_area / fruit_area if fruit_area > 0 else 0.0
    occ_overripe = overripe_area / fruit_area if fruit_area > 0 else 0.0

    contours, _ = cv2.findContours(
        fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # min_area 以上の輪郭のbboxを候補にする
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

    # 分割された房を1つのbboxに統合して丸ごと切る（本番運用でも常時適用の固定挙動）。
    #   双子/奇形果や、果柄除去の開処理で割れた本体を1枚に戻す。1画像1果実が前提のため、
    #   複数の果実が写る場合はそれらも1つの枠に統合される点に注意。
    if boxes:
        x0 = min(x        for (x, y, w, h) in boxes)
        y0 = min(y        for (x, y, w, h) in boxes)
        x1 = max(x + w    for (x, y, w, h) in boxes)
        y1 = max(y + h    for (x, y, w, h) in boxes)
        rects = [(x0, y0, x1 - x0, y1 - y0)]
    else:
        rects = []

    return occ_unripe, occ_overripe, len(rects) > 0, rects


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
) -> QPixmap:
    """3クラス判定で枠色を決定し、左上に未熟占有率・過熟占有率を描画したサムネイルを返す。
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

    annotated = thumb.copy()
    for (rx, ry, rw, rh) in rects:
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 0), 1)

    # デバッグ表示: マスクの3段階（生→虚像除去→果柄除去）＋未熟/過熟マスクの輪郭を色分けで重ねる
    if debug and fruit_params is not None:
        hsv_t  = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
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
