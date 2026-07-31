"""設定の既定値・定数・JSON入出力（本体: 未熟マスク・過熟マスク占有率モデル）。"""

import json
from pathlib import Path

from PySide6.QtGui import QColor

# ==========================================================
# 設定
# ==========================================================
TRAIN_DIR   = r"E:\DCRpcから持ってきたもの\model作成用imageset\all_true"
THUMB_SIZE  = 112
GRID_COLS   = 7
DEBOUNCE_MS = 300

_TRAIN_PATH = Path(TRAIN_DIR)
HEALTHY_DIR  = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_ripeness_healthy"
UNRIPE_DIR   = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_ripeness_unripe"
OVERRIPE_DIR = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_ripeness_overripe"

_JSON_DIR    = Path(__file__).parent.parent / "json"
_CONFIG_PATH = _JSON_DIR / "hsv_ripeness_config.json"

# 果実全体マスクのデフォルト（黄〜橙〜赤を広くカバー）
DEFAULT_FRUIT_HSV = {
    "lower1": [0,   100, 100],
    "upper1": [32,  255, 255],
    "lower2": [160, 100, 100],
    "upper2": [180, 255, 255],
}

# 未熟マスクのデフォルト（黄〜緑の未熟色）。
#   H は 0/180 をまたがない範囲なので、GUI上は色相バー1本のみ表示する
#   （HsvSliderGroup(single_hue=True)。lower2/upper2 は get_params() が lower1/upper1 を
#   自動複製するので、mask_from_hsv 側の2帯構成はそのまま使える）。
DEFAULT_UNRIPE_HSV = {
    "lower1": [20, 60, 60],
    "upper1": [75, 255, 255],
    "lower2": [20, 60, 60],
    "upper2": [75, 255, 255],
}

# 過熟マスクのデフォルト（暗く赤黒ずんだ部分のみ）。
#   V上限を絞って「暗さ」を表現し、このマスクのHSV範囲そのものが
#   「過熟の色相・明度境界」を兼ねる（別途の色相・明度閾値は持たない）。
#   V上限は果実マスクのV下限(100)より高く取り、両マスクのV範囲が重なるようにする
#   （重ならないと過熟マスク∩果実マスクが常に空になり、過熟占有率が常に0になってしまう）。
DEFAULT_OVERRIPE_HSV = {
    "lower1": [0,   10, 30],
    "upper1": [9,   255, 160],
    "lower2": [170, 10, 30],
    "upper2": [180, 255, 160],
}

# 未熟確定の占有率境界（%）: 未熟マスク∩果実マスクの占有率(occ_unripe)がこれ以上なら未熟。
DEFAULT_OCC_UNRIPE_CONFIRM = 25
# 過熟確定の占有率境界（%）: 過熟マスク∩果実マスクの占有率(occ_overripe)がこれ以上なら過熟。
DEFAULT_OCC_OVERRIPE_CONFIRM = 50
DEFAULT_MIN_AREA   = 300  # 果実マスクの最小面積（サムネイル換算px²）
# 果柄除去（開処理）半径（サムネイル換算px）: erode→dilate で細い果柄を切り離す。
#   0 = 無効。太い本体は残り、半径より細い果柄だけが消える。値を上げるほど太い果柄まで除去。
#   保存時は元画像スケールへ換算する（min_area と同様）。
DEFAULT_STEM_OPEN = 6
# 虚像除去の彩度範囲（0〜255・HSVマスクと同じ左右ハンドルで指定）: アクリル板の反射(虚像)は
#   実体より明らかに淡い(低彩度)。S∈[下限,上限] の画素をマスクから外して虚像を落とす。
#   範囲が全域(0〜255)なら無効。典型的には低彩度側([0, 100]など)を指定する。
#   縦並び双子果の房境界は高彩度なので分断されない（幾何くびれ方式と違い衝突しない）。
#   彩度による判定なのでスケール非依存。
DEFAULT_REFLECT_SAT_LO = 0
DEFAULT_REFLECT_SAT_HI = 255
# 虚像除去の明度範囲（0〜255・試験導入）: 反射(虚像)は白飛びで明るいことが多いため、
#   V∈[下限,上限] の画素もマスクから外す。範囲が全域(0〜255)なら無効。
#   典型的には高明度側([200, 255]など)を指定する。彩度範囲と両方有効なときは
#   両条件を同時に満たす画素だけを虚像として除去する（remove_reflection_sat参照）。
DEFAULT_REFLECT_V_LO = 0
DEFAULT_REFLECT_V_HI = 255

COLOR_HEALTHY  = QColor(0,  210,  0)
COLOR_UNRIPE   = QColor(230, 140,  0)
COLOR_OVERRIPE = QColor(168,  70, 200)   # 紫（過熟）
COLOR_NONE     = QColor(210,   0,  0)

# 保存先ルートの対応（target → ディレクトリ）
_DEST_ROOTS = {"healthy": HEALTHY_DIR, "unripe": UNRIPE_DIR, "overripe": OVERRIPE_DIR}


# ==========================================================
# JSON 入出力
# ==========================================================
def load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        # 旧フォーマット（新キー無し）でも動くよう既定値で補完する
        cfg.setdefault("unripe",     {k: list(v) for k, v in DEFAULT_UNRIPE_HSV.items()})
        cfg.setdefault("overripe",   {k: list(v) for k, v in DEFAULT_OVERRIPE_HSV.items()})
        cfg.setdefault("occ_unripe_confirm",   DEFAULT_OCC_UNRIPE_CONFIRM)
        cfg.setdefault("occ_overripe_confirm", DEFAULT_OCC_OVERRIPE_CONFIRM)
        cfg.setdefault("stem_open",    DEFAULT_STEM_OPEN)
        # 旧キー（reflect_sat=彩度下限のみ, reflect_v_ceil=明度上限のみ）から
        #   新キー（左右ハンドルの範囲指定）へ移行する。旧値の挙動をできるだけ再現する:
        #   彩度は「低彩度側を虚像とみなす」ので [0, 旧floor] に、
        #   明度は「高明度側を虚像とみなす」ので [旧ceil, 255] に対応させる。
        if "reflect_sat_lo" not in cfg or "reflect_sat_hi" not in cfg:
            old_sat = cfg.get("reflect_sat", 0)
            if old_sat > 0:
                cfg["reflect_sat_lo"], cfg["reflect_sat_hi"] = 0, old_sat
            else:
                cfg["reflect_sat_lo"], cfg["reflect_sat_hi"] = DEFAULT_REFLECT_SAT_LO, DEFAULT_REFLECT_SAT_HI
        if "reflect_v_lo" not in cfg or "reflect_v_hi" not in cfg:
            old_vceil = cfg.get("reflect_v_ceil", 0)
            if old_vceil > 0:
                cfg["reflect_v_lo"], cfg["reflect_v_hi"] = old_vceil, 255
            else:
                cfg["reflect_v_lo"], cfg["reflect_v_hi"] = DEFAULT_REFLECT_V_LO, DEFAULT_REFLECT_V_HI
        cfg.pop("reflect_sat", None)
        cfg.pop("reflect_v_ceil", None)
        # 旧モデル（赤色/深赤マスク・threshold）の残存キーは不要なので取り除く
        cfg.pop("red", None)
        cfg.pop("deepred", None)
        cfg.pop("threshold", None)
        return cfg
    return {
        "fruit":        {k: list(v) for k, v in DEFAULT_FRUIT_HSV.items()},
        "unripe":       {k: list(v) for k, v in DEFAULT_UNRIPE_HSV.items()},
        "overripe":     {k: list(v) for k, v in DEFAULT_OVERRIPE_HSV.items()},
        "occ_unripe_confirm":   DEFAULT_OCC_UNRIPE_CONFIRM,
        "occ_overripe_confirm": DEFAULT_OCC_OVERRIPE_CONFIRM,
        "min_area":     DEFAULT_MIN_AREA,
        "stem_open":    DEFAULT_STEM_OPEN,
        "reflect_sat_lo": DEFAULT_REFLECT_SAT_LO,
        "reflect_sat_hi": DEFAULT_REFLECT_SAT_HI,
        "reflect_v_lo":   DEFAULT_REFLECT_V_LO,
        "reflect_v_hi":   DEFAULT_REFLECT_V_HI,
    }


def save_config(cfg: dict):
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
