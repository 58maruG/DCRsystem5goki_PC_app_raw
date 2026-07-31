"""日本語パス対応の画像入出力ユーティリティ（3バリアント共通）。"""

from pathlib import Path

import cv2
import numpy as np


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
