from __future__ import annotations
import cv2
import numpy as np
import datetime
import os
import time
import json
import queue
import threading
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers.byte_tracker import BYTETracker

import log_config
log = log_config.get_logger("yolo")

# get_target_info はフレームごとに呼ばれるため、初回ロード時のみJSONを読む
_logged_hsv_paths: set[str] = set()
_hsv_config_cache: dict[str, dict] = {}

# ================================================
# モデル・入力設定
# ================================================
USE_CROP          = False
CENTER_THRESHOLD_X = 100

# ================================================
# 推論ゲート（中心帯）設定
# ================================================
# YOLO推論ゲートの「左右2本の縦線」の、ROI中心からの距離【絶対ピクセル・カメラ別】。
#   左線 x = w/2 - half / 右線 x = w/2 + half（w は各カメラのROI幅 = pfsのWidth）。
#   この2本が「両方とも」HSVマスク(最大ブロブ)に被る間だけ推論する
#   （＝サクランボが帯を跨いで2本を内部に含むとき）。GUIにはこの2本を縦線で表示する。
#   ※ ROI幅がカメラごとに異なる（cam_pfs参照）。割合だと線間隔・推論タイミングがカメラ間で
#     ずれるため、絶対pxでカメラ別に持つ。GUIの縦線を見て個別に調整する。
#   2本を近づけたい（跨ぎやすく＝推論が始まりやすい）→ 値を小さく / 遠ざけたい → 大きく。
#   初期値は従来比率(ROI幅×0.1)相当。実機の見え方に合わせて詰めること。
BAND_HALF_PX = {
    'cam_top':     64,   # ROI幅 640
    'cam_under':   64,   # ROI幅 640
    'cam_inside':  56,   # ROI幅 560
    'cam_outside': 50,   # ROI幅 500
}
DEFAULT_BAND_HALF_PX = 56   # 上記に無いカメラ用のフォールバック

#MODEL_PATH = "Trained_Models/v2_11s.pt"
MODEL_PATH   = "Trained_Models/v4_11s.pt"
YOLO_IMG_SIZE = 640
CONF_THRESHOLD = 0.5

# ================================================
# クラス別信頼度閾値
# ================================================
# strict閾値:不良として採用する最小信頼度
# 取りこぼしが気になる場合はさらに下げ、良品の誤排出が増える場合は上げる。
# クラス別 strict 閾値。判定ロジックの全段階と GUI表示フィルタ(class_breakdown)で共用する。
#   ここに無いクラス(カビ・灰星病など)は閾値なし＝常に採用・表示。
STRICT_THRESHOLDS = {
    "stemcrack":    0.8,
    "crack":        0.8,
    "birddamage":   0.8,
    "twin":         0.9,
    "blacktwin":    0.9,
    "malformation": 0.9,
    "unripe":       0.9,
}

# ================================================
# マルチカメラ確定ロジック設定
# ================================================
# twin/malformation/blacktwin: この台数以上の「異なるカメラ」で検出されたら、
#   全体の最大信頼度によらずそのクラスを確定する（同一カメラの複数フレームは1カウント）。
MULTI_CAM_MIN = 2
# unripe を確定するのに必要なカメラ台数
UNRIPE_MIN_CAMS = 3

# ================================================
# セグメンテーション（個体区切り）設定
# ================================================
# 不在タイムアウト（秒）: いずれのカメラもこの時間サクランボを検出しなければ
#   「1個分が通過し終わった」とみなして確定する（出口ヒステリシス）。
#   大きすぎる → 近接した2個が1個に統合される / 小さすぎる → 1個が複数IDに分割される
EMPTY_TIMEOUT_SEC = 0.5
# 最小可視時間（秒）: 個体が確定対象として「本物」と認められるための最小の可視継続。
#   これ未満しか見えず、かつYOLO検出も無かった瞬間的なノイズ blip は破棄し、
#   幽霊ID・黒タイルの量産を防ぐ（入口ヒステリシス）。
#   ※ YOLO検出が1度でもあれば、可視時間に依らず本物として確定する。
MIN_VISIBLE_SEC = 0.12

# ================================================
# ファイル保存設定
# ================================================
SAVE_DIR_IMG      = "evaluated_images"
SAVE_DIR_TRAINING = "training_images"
MIN_TRAINING_AREA = 10000   # 学習用画像として保存する最小検出面積（小さすぎるフレームを除外）
FPS               = 20.0

# ================================================
# 追跡・可視化設定
# ================================================
# ByteTrack に渡す前段検出の信頼度しきい値。
# 低めにしてトラッカーへ多くの情報を渡し、カルマン予測の精度を上げる。
# ラベル採用の最終判定は CONF_THRESHOLD で行う（2段階フィルタ）。
PREDICT_CONF = 0.1
CAM_NAMES    = ['cam_top', 'cam_under', 'cam_inside', 'cam_outside']
COLORS = {
    "birddamage":  (225, 105,  65),
    "healthy":     (255, 255, 255),
    "mold":        (128,   0, 128),
    "stemcrack":   (255,   0,   0),
    "twin":        (  0,   0, 255),
    "unripe":      (  0, 255, 255),
    "malformation":(  0,  69, 255),
    "crack":       (255, 191,   0),
    "wilt":        ( 42, 107, 142),
    "suturecrack": (170, 178,  32),
    "brownrot":    ( 45,  82, 160),
    "blacktwin":   ( 79,  79,  47),
    "kasure":      (131, 180, 212),
}


# ================================================
# 判定結果データクラス
# ================================================
class YoloResult:
    def __init__(self, obj_id: int, label_name: str,
                 confidence: float, cam_name: str) -> None:
        self.id           = obj_id
        self.label_name   = label_name
        self.confidence   = confidence
        self.cam_name     = cam_name

        # 個体確定時に _finalize_object が付与するサイクル集計（cycle ログ用）。
        # 既定値を持たせ、未確定の中間結果でも属性参照で落ちないようにする。
        self.num_detections     = 0      # この個体の総検出数（全フレーム・全カメラ）
        self.conf_max           = None   # 確定クラスの信頼度 最大
        self.conf_min           = None   # 確定クラスの信頼度 最小
        self.conf_avg           = None   # 確定クラスの信頼度 平均
        self.infer_avg_ms       = None   # この個体の推論時間平均(ms)
        self.preproc_ms         = None   # 前処理（クロップ・リサイズ）時間平均(ms)
        self.postproc_ms        = None   # 後処理（アノテーション・parse）時間平均(ms)
        self.capture_latency_ms = None   # カメラフレーム取得時間の平均(ms)
        self.frame_dropped      = None   # この個体の通過中にドロップしたフレーム数（全カメラ合計）
        self.hsv_pass           = None   # YOLO検出が1度でもあったか（1=あり/0=なし）
        self.hsv_mask_ratio     = None   # HSVマスク面積比の平均（0〜1）
        self.yolo_no_det_flag   = None   # HSV通過・YOLO無検出フラグ（1=HSV有でYOLO未検出 / 0=正常検出）
        # この個体で検出された各クラスの「最大信頼度」を信頼度降順で並べたリスト。
        #   要素は (label_name, conf_max) のタプル。GUIの複数クラス表示が読む。
        self.class_breakdown    = []


# ================================================
# 画像保存・CSV出力クラス
# ================================================
class OutputLogger:
    def __init__(self, dcr=None) -> None:
        self.dcr = dcr

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 実行ごとのサブフォルダ（IDの重複防止）
        self.run_img_dir = os.path.join(SAVE_DIR_IMG, timestamp)
        # os.makedirs(self.run_img_dir, exist_ok=True)  # 画像保存一時停止 ← 再開時はコメントを外す

        # 学習用画像ディレクトリ（YOLOクラス名別・実行をまたいで蓄積）
        self.valid_cam_names = ('cam_top', 'cam_under', 'cam_inside', 'cam_outside')
        # os.makedirs(SAVE_DIR_TRAINING, exist_ok=True)  # 画像保存一時停止 ← 再開時はコメントを外す

    def write_csv(self, obj_id: int, detections: list, final_label: str) -> None:
        """1個体（1 ID）の検出履歴を多ラベル long 形式で新ロガーへ送る。
        検出されたクラスごとに1件を出力し、同一クラスの複数検出は
        信頼度の min/max/平均と件数 n に集約する（異なるクラスは分けて記入）。
        final_label は仕分けで採用された確定クラス名。一致する行の is_final を1にする。"""
        if self.dcr is None:
            return
        # クラスごとに信頼度を集約（"None"＝未検出は除外）
        by_class: dict[str, list[float]] = {}
        for d in detections:
            if d.label_name == "None":
                continue
            by_class.setdefault(d.label_name, []).append(d.confidence)
        if not by_class:
            return

        items = []
        for label, confs in by_class.items():
            n = len(confs)
            items.append({
                "class":          label,
                "conf_ave":       round(sum(confs) / n, 2),
                "conf_min":       round(min(confs), 2),
                "conf_max":       round(max(confs), 2),
                "num_detections": n,
                "final_flag":     1 if label == final_label else 0,
            })
        self.dcr.detections(obj_id, items)

    def write_training_image(self, cam_name: str, frame, label_name: str | None = None) -> None:
        """学習用の生フレームをYOLOクラス名別ディレクトリに保存する。
        保存先は training_images/<クラス名>/ で、ファイル名は「推論クラス名_撮影時刻_カメラ名」。
        推論できなかった場合（label_name が None / 空 / "None"）はクラス名を NoClass とする。"""
        # 画像保存一時停止 ← 再開時はコメントを外す
        # if cam_name in self.valid_cam_names:
        #     cls     = label_name if label_name and label_name != "None" else "NoClass"
        #     cls_dir = os.path.join(SAVE_DIR_TRAINING, cls)
        #     os.makedirs(cls_dir, exist_ok=True)
        #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        #     filename  = f"{timestamp}_{cls}_{cam_name}.jpg"
        #     filepath  = os.path.join(cls_dir, filename)
        #     cv2.imwrite(filepath, frame)

    @staticmethod
    def _placeholder_tile(cam_name: str):
        """合格フレームも直近フレームも無いカメラ用の代替タイル。
        純黒(np.zeros)だと象限が完全に潰れて原因が分からないため、
        グレー背景＋カメラ名＋"NO SIGNAL"を描いて「映像が無かった」ことを明示する。"""
        tile = np.full((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), 60, dtype=np.uint8)
        for text, y in ((cam_name, YOLO_IMG_SIZE // 2 - 20), ("NO SIGNAL", YOLO_IMG_SIZE // 2 + 30)):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(tile, text, ((YOLO_IMG_SIZE - tw) // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        return tile

    def write_evaluated_image(self, frames_dict: dict, obj_id: int,
                               label_name: str, fallback_frames: dict | None = None,
                               last_frames: dict | None = None) -> None:
        """4カメラのアノテーション済みフレームを2×2タイルで1枚保存する。
        各象限は次の優先順で埋める（純黒を出さないことで「黒タイル」問題を回避）:
          1. タイル用最良フレーム（YOLO検出＞HSV） (frames_dict)
          2. HSVで捉えた学習用の生フレーム          (fallback_frames)
          3. そのカメラの直近フレーム                (last_frames)
          4. グレーの "NO SIGNAL" プレースホルダ"""
        # 画像保存一時停止 ← 再開時はコメントを外す
        # cam_order = ['cam_inside', 'cam_outside', 'cam_under', 'cam_top']
        # tiles = []
        # for cam in cam_order:
        #     if frames_dict and cam in frames_dict:
        #         tiles.append(cv2.resize(frames_dict[cam], (YOLO_IMG_SIZE, YOLO_IMG_SIZE)))
        #     elif fallback_frames and cam in fallback_frames:
        #         tiles.append(cv2.resize(fallback_frames[cam], (YOLO_IMG_SIZE, YOLO_IMG_SIZE)))
        #     elif last_frames and last_frames.get(cam) is not None:
        #         tiles.append(cv2.resize(last_frames[cam], (YOLO_IMG_SIZE, YOLO_IMG_SIZE)))
        #     else:
        #         tiles.append(self._placeholder_tile(cam))
        # tile      = np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], tiles[3]))))
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # filename  = f"id{obj_id:04d}_{label_name}_{timestamp}.jpg"
        # filepath  = os.path.join(self.run_img_dir, filename)
        # cv2.imwrite(filepath, tile)


# ================================================
# 画像処理ユーティリティクラス
# ================================================
class ImageProcessor:
    @staticmethod
    def get_target_info(frame, cam_name: str | None = None) -> dict | None:
        # ここはYOLO判定ではなく「サクランボが視野内にいるか（存在）」を追うためのHSV赤マスク
        #
        # ── HSV 設定の切り替え方 ──────────────────────────────────────────
        # [A] カメラ別JSON（hsv_calibration.py で生成）を優先する ← 現在の設定
        #       hsv_config_{cam_name}.json → hsv_config.json → デフォルト値
        #       の順にフォールバックするので、カメラ別ファイルが無くても動く。
        # ──────────────────────────────────────────────────────────────────

        # カメラ別JSON
        search_paths = []
        if cam_name:
            search_paths.append(os.path.join("json", f"hsv_config_{cam_name}.json"))

        cfg = None
        for path in search_paths:
            if path in _hsv_config_cache:
                cfg = _hsv_config_cache[path]
                break
            if os.path.exists(path):
                with open(path, 'r') as f:
                    cfg = json.load(f)
                _hsv_config_cache[path] = cfg
                if path not in _logged_hsv_paths:
                    log.info("HSV設定を読み込みました: %s", path)
                    _logged_hsv_paths.add(path)
                break

        if cfg:
            lower_red1, upper_red1 = np.array(cfg['lower1']), np.array(cfg['upper1'])
            lower_red2, upper_red2 = np.array(cfg['lower2']), np.array(cfg['upper2'])
        else:
            # デフォルト値（JSONファイルが一切存在しない場合）
            lower_red1, upper_red1 = np.array([0, 100, 100]),   np.array([32, 255, 255])
            lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2),
            cv2.MORPH_CLOSE, kernel, iterations=2
        )
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return None
        max_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        if stats[max_index, cv2.CC_STAT_AREA] < 500:
            return None
        h, w = frame.shape[:2]
        s = stats[max_index]
        if (s[0] <= 5) or (s[1] <= 5) or ((s[0] + s[2]) >= (w - 5)) or ((s[1] + s[3]) >= (h - 5)):
            return None
        return {'mx': int(centroids[max_index][0]), 'my': int(centroids[max_index][1]),
                'area': s[4], 'stat': s,
                'labels': labels, 'max_index': int(max_index)}

    @staticmethod
    def dynamic_crop(frame, target: dict):
        if not USE_CROP:
            return frame
        h, w  = frame.shape[:2]
        size  = min(int(max(target['stat'][2], target['stat'][3]) * 1.5), w, h)
        x1 = max(0, target['mx'] - size // 2)
        y1 = max(0, target['my'] - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        if x2 == w: x1 = max(0, w - size)
        if y2 == h: y1 = max(0, h - size)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def draw_inference_band(frame, cam_name, native_width, color=(0, 200, 255), thickness=2):
        """推論ゲートの左右2本の縦線を frame に描画する（GUI可視化用）。
        ゲートは native_width(=ROI幅) 上の絶対px(BAND_HALF_PX)で判定するため、
        表示フレームが640へリサイズ済みでも比率に直して正しい位置へ描く。
        color は frame の色空間に合わせる（RGB画像へ描くなら RGB 並びで渡す）。"""
        h, w = frame.shape[:2]
        half = BAND_HALF_PX.get(cam_name, DEFAULT_BAND_HALF_PX)
        frac = (half / native_width) if native_width else 0.0
        xl = min(max(int(round((0.5 - frac) * w)), 0), w - 1)
        xr = min(max(int(round((0.5 + frac) * w)), 0), w - 1)
        cv2.line(frame, (xl, 0), (xl, h - 1), color, thickness)
        cv2.line(frame, (xr, 0), (xr, h - 1), color, thickness)
        return frame


# ================================================
# YOLO検出クラス
# ================================================
class YoloDetector:
    def __init__(self, model_path: str = MODEL_PATH, dcr=None, cameras=None) -> None:
        log.info("YOLOモデル %s をロード中...", model_path)
        self.dcr     = dcr       # データログ（cycle は main、detections は logger 経由）
        self.cameras = cameras   # list[CameraController]（サイクル統計取得用）
        self.model   = YOLO(model_path)
        dummy_img    = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        self.logger  = OutputLogger(dcr=dcr)

        # カメラごとに独立した ByteTracker（カメラ間で追跡状態が混ざらないよう分離）
        # track_buffer を FPS に合わせてスケール（旧 frame_rate 引数相当の補正）
        _cfg_dict = YAML.load(check_yaml("bytetrack.yaml"))
        _cfg_dict["track_buffer"] = int(_cfg_dict["track_buffer"] * FPS / 30.0)
        tracker_cfg  = IterableSimpleNamespace(**_cfg_dict)
        self.trackers = {cam: BYTETracker(tracker_cfg) for cam in CAM_NAMES}

        self.EMPTY_TIMEOUT_SEC = EMPTY_TIMEOUT_SEC
        self.MIN_VISIBLE_SEC   = MIN_VISIBLE_SEC

        self.current_cherry_id = 1
        self.last_seen_time    = time.monotonic()  # いずれかのカメラが最後に検出した時刻（不在判定用）

        # カメラごとの直近フレーム（合成タイルの最終フォールバック用・個体をまたいで保持）
        self.last_frame_per_cam = {c: None for c in CAM_NAMES}

        # 現在追跡中の1個体の状態（_reset_object で初期化）
        self._obj_generation = 0   # _reset_object のたびに増加。古い推論結果を破棄する目印
        self._reset_object()

        # ── 推論ワーカースレッド ──────────────────────────────────────
        # model.predict だけを別スレッドで実行し、GUIスレッドのブロックを防ぐ。
        # キューのサイズを小さく保ち、古いフレームは捨てて遅延蓄積を防ぐ。
        self._infer_queue    = queue.Queue(maxsize=4)
        self._result_queues  = {cam: queue.Queue(maxsize=2) for cam in CAM_NAMES}
        self._infer_running  = True
        self._infer_thread   = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()

    def model_precision(self) -> str:
        """ロード済みモデルの実際の重み精度を返す（startup ログ用）。
        half を明示していなくても torch の dtype から判定する。取得不可なら空。"""
        try:
            import torch
            dt = next(self.model.model.parameters()).dtype
            return {torch.float16: "fp16", torch.float32: "fp32",
                    torch.bfloat16: "bf16"}.get(dt, str(dt).replace("torch.", ""))
        except Exception:
            return ""

    # ------------------------------------------------------
    # 個体状態のリセット／更新ヘルパー
    # ------------------------------------------------------
    def _reset_object(self) -> None:
        self._obj_generation   += 1       # 古い推論結果を無効化するための世代番号
        self.obj_active         = False   # 追跡中の個体があるか
        self.obj_first_seen     = None    # 現個体が最初に見えた時刻
        self.obj_last_seen      = None    # 現個体が最後に見えた時刻
        self.obj_has_detection  = False   # 現個体でYOLO検出があったか
        self.obj_detections     = []      # 現個体のYoloResult履歴（全カメラ）
        self.obj_cam_tile       = {}      # タイル用: {cam: {'frame','priority','metric'}}
        self.obj_cam_train      = {}      # 学習用:   {cam: {'min_dist','frame'}}（生フレーム）
        self.obj_infer_ms_sum   = 0.0     # 現個体の推論時間の合計(ms)
        self.obj_infer_count    = 0       # 現個体で推論したフレーム数
        self.obj_preproc_ms_sum  = 0.0    # 前処理時間の合計(ms)
        self.obj_preproc_count   = 0
        self.obj_postproc_ms_sum = 0.0    # 後処理時間の合計(ms)
        self.obj_postproc_count  = 0
        self.obj_hsv_area_sum    = 0.0    # HSVマスク面積比の合計
        self.obj_hsv_area_count  = 0

    def _update_cam_tile(self, cam_name: str, frame,
                          priority: int, metric: float) -> None:
        """タイル用フレーム保持を一本化して更新する。
        priority: 2=YOLO検出 / 1=HSVのみ。 metric: 同priority内の優劣（大きいほど良い）。
          - YOLO検出: metric = confidence（高いほど良い）
          - HSVのみ : metric = -center_dist（中心に近いほど良い）"""
        cur = self.obj_cam_tile.get(cam_name)
        if cur is None or priority > cur['priority'] or (priority == cur['priority'] and metric > cur['metric']):
            self.obj_cam_tile[cam_name] = {'frame': frame.copy(), 'priority': priority, 'metric': metric}

    def _finalize_object(self) -> YoloResult | None:
        """現個体を確定（または破棄）する。確定したら YoloResult を返す。
        入口ヒステリシス: 可視時間が MIN_VISIBLE_SEC 未満かつYOLO検出も無い blip は破棄しIDを進めない。"""
        result = None
        if self.obj_first_seen is not None:
            visible_dur = self.obj_last_seen - self.obj_first_seen
            confirmed   = self.obj_has_detection or (visible_dur >= self.MIN_VISIBLE_SEC)
            if confirmed:
                if self.obj_detections:
                    best = self._resolve_best_result(self.obj_detections)
                    if best:
                        # long形式CSV: このIDの全検出をクラス別に集約して書き出す。
                        self.logger.write_csv(best.id, self.obj_detections, best.label_name)
                        # cycle ログ用の集計を best に添える（main が process_final_result で読む）
                        self._attach_cycle_stats(best)
                        result = best
                        cid    = best.id
                        # 評価済みタイル: タイル用最良フレーム→学習用生フレーム→直近→プレースホルダ で4枚合成
                        tile_frames  = {cam: d['frame'] for cam, d in self.obj_cam_tile.items()}
                        train_frames = {cam: d['frame'] for cam, d in self.obj_cam_train.items()}
                        self.logger.write_evaluated_image(tile_frames, cid, best.label_name,
                                                          train_frames, self.last_frame_per_cam)
                        # 学習用画像: 各カメラを「そのカメラの最高信頼度クラス」フォルダに保存
                        for cam, d in self.obj_cam_train.items():
                            self.logger.write_training_image(cam, d['frame'], d['label'])
                    self.current_cherry_id += 1
                else:
                    # 本物だがYOLO未検出 → 学習用保存 + cycle ログに yolo_no_det_flag=1 で記録
                    for cam, d in self.obj_cam_train.items():
                        self.logger.write_training_image(cam, d['frame'], d['label'])
                    no_det = YoloResult(self.current_cherry_id, "None", 0.0, "")
                    self._attach_cycle_stats(no_det, yolo_no_det=1)
                    result = no_det
                    self.current_cherry_id += 1
            # confirmed でない（短すぎる blip）→ 破棄し、IDは進めない
        self._reset_object()
        return result

    def _attach_cycle_stats(self, best: YoloResult, yolo_no_det: int = 0) -> None:
        """確定個体の集計（検出数・確定クラスの信頼度統計・推論時間平均）を best に付与する。
        main 側の process_final_result がこれを読んで dcr.cycle(...) に渡す。
        yolo_no_det=1 のときは HSV通過・YOLO無検出ケース。"""
        best.yolo_no_det_flag = yolo_no_det
        confs = [d.confidence for d in self.obj_detections if d.label_name == best.label_name]
        best.num_detections = len([d for d in self.obj_detections if d.label_name != "None"])
        if confs:
            best.conf_max = round(max(confs), 3)
            best.conf_min = round(min(confs), 3)
            best.conf_avg = round(sum(confs) / len(confs), 3)

        # クラス別の最大信頼度を集計し、信頼度降順で best に添える（GUIの複数クラス表示用）。
        #   "None"（未検出）は除外。write_csv の by_class と同じ集計方針。
        by_class: dict[str, float] = {}
        for d in self.obj_detections:
            if d.label_name == "None":
                continue
            if d.label_name not in by_class or d.confidence > by_class[d.label_name]:
                by_class[d.label_name] = d.confidence
        breakdown = sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)
        # strict閾値未満のクラスはGUIに表示しない。ただし確定クラス(best.label_name)は
        #   閾値に関わらず残す（有効検出ゼロ時に検出信頼度で確定したクラスを消さないため）。
        #   閾値の無いクラス(カビ・灰星病など)は従来どおり常に表示。
        best.class_breakdown = [
            (lbl, conf) for lbl, conf in breakdown
            if lbl == best.label_name or conf >= STRICT_THRESHOLDS.get(lbl, 0.0)
        ]
        if self.obj_infer_count > 0:
            best.infer_avg_ms = round(self.obj_infer_ms_sum / self.obj_infer_count, 2)
        if self.obj_preproc_count > 0:
            best.preproc_ms = round(self.obj_preproc_ms_sum / self.obj_preproc_count, 2)
        if self.obj_postproc_count > 0:
            best.postproc_ms = round(self.obj_postproc_ms_sum / self.obj_postproc_count, 2)
        if self.obj_hsv_area_count > 0:
            best.hsv_mask_ratio = round(self.obj_hsv_area_sum / self.obj_hsv_area_count, 4)
        best.hsv_pass = 1 if self.obj_has_detection else 0
        # カメラのサイクル統計を集約（cameras が渡されている場合のみ）
        if self.cameras:
            total_dropped = 0
            latencies     = []
            for cam in self.cameras:
                stats = cam.get_cycle_stats()
                total_dropped += stats['frame_dropped']
                if stats['capture_latency_ms'] is not None:
                    latencies.append(stats['capture_latency_ms'])
            best.frame_dropped = total_dropped
            if latencies:
                best.capture_latency_ms = round(sum(latencies) / len(latencies), 2)

    def _resolve_best_result(self, detections: list) -> YoloResult | None:
        """
        全カメラの履歴から最終判定を決定する。
        受け取る detections は蓄積時点で STRICT_THRESHOLDS 済み。

        (1) 「複数カメラ一致による確定ルール」を判定する。
            MULTI_CAM_MIN 台以上で成立すれば、そのクラスを確定する。
        (2) 成立しなければフォールバックで決める。優先順は
            「不良(未熟含む) > 健全」。ただし残りが未熟と健全のみの場合は
            検出カメラ数の過半数で決め、同数なら信頼度が高い方を採用する。
        """
        if not detections:
            return None

        # --- クラスごとに「検出した異なるカメラ数」を集計（同一カメラの複数フレームは1カウント）---
        cams_per_label: dict[str, set] = {}
        for d in detections:
            cams_per_label.setdefault(d.label_name, set()).add(d.cam_name)

        def cam_count(label: str) -> int:
            return len(cams_per_label.get(label, ()))

        def best_of(label: str) -> YoloResult | None:
            cands = [d for d in detections if d.label_name == label]
            return max(cands, key=lambda x: x.confidence) if cands else None

        # ============================================================
        # (1) 確定ルール: 指定クラスが複数カメラ(閾値以上)で検出されたら確定
        # ============================================================
        # twin / malformation / blacktwin: MULTI_CAM_MIN 台以上の異なるカメラで検出 → 確定。
        confirmed = [best_of(lbl) for lbl in ("twin", "malformation", "blacktwin")
                     if cam_count(lbl) >= MULTI_CAM_MIN]
        if confirmed:
            return max(confirmed, key=lambda x: x.confidence)

        # unripe: UNRIPE_MIN_CAMS 台以上で確定
        if cam_count("unripe") >= UNRIPE_MIN_CAMS:
            return best_of("unripe")

        # ============================================================
        # (2) フォールバック: 確定ルール非成立時は最大信頼度ベースで決める
        # ============================================================
        #   detections は既に閾値以上のみなので、ここでの個別閾値チェックは不要。
        #   未熟・健全以外は全て「不良(other_damaged)」として扱う。
        healthy_list       = [d for d in detections if d.label_name == "healthy"]
        unripe_list        = [d for d in detections if d.label_name == "unripe"]
        other_damaged_list = [d for d in detections
                              if d.label_name not in ("healthy", "unripe")]

        # ------------------------------------------------------------
        # 優先順位の決定: 不良(未熟含む) > 健全
        #   1. 未熟以外の不良があれば最優先（最高信頼度）。
        #   2. 残りが未熟と健全のみなら、検出した異なるカメラ数の多い方（過半数）を採用。
        #      同数なら信頼度が高い方を採用する。
        # ------------------------------------------------------------
        if other_damaged_list:
            return max(other_damaged_list, key=lambda x: x.confidence)

        best_unripe  = max(unripe_list,   key=lambda x: x.confidence) if unripe_list  else None
        best_healthy = max(healthy_list,  key=lambda x: x.confidence) if healthy_list else None
        if best_unripe and best_healthy:
            unripe_cams  = len({d.cam_name for d in unripe_list})
            healthy_cams = len({d.cam_name for d in healthy_list})
            if unripe_cams != healthy_cams:
                return best_unripe if unripe_cams > healthy_cams else best_healthy
            # 同数 → 信頼度が高い方を採用
            return best_unripe if best_unripe.confidence >= best_healthy.confidence else best_healthy
        if best_unripe:
            return best_unripe
        if best_healthy:
            return best_healthy

        # ガード: ここに到達するのは other_damaged/unripe/healthy が全て空のときだが、
        #   detections は閾値以上の非空リストなので通常は手前で return される。
        return max(detections, key=lambda x: x.confidence)

    # ------------------------------------------------------
    # 推論ワーカー（バックグラウンドスレッド）
    # ------------------------------------------------------
    def _inference_worker(self) -> None:
        """model.predict だけを別スレッドで実行し、結果をカメラ別キューへ返す。
        GUIスレッドのQTimerをブロックしないことが目的。ByteTrackerは含まない。"""
        while self._infer_running:
            try:
                item = self._infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            cam_name, img, ctx = item
            t0 = time.perf_counter()
            try:
                results = self.model.predict(img, conf=PREDICT_CONF, verbose=False)
                det     = results[0].boxes.cpu().numpy()
            except Exception as e:
                log.error("推論ワーカー例外: %s", e)
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0
            try:
                self._result_queues[cam_name].put_nowait((det, img, infer_ms, ctx))
            except queue.Full:
                pass  # 結果キューが満杯なら古い結果が先に消化されるまで破棄

    def _apply_inference(self, cam_name: str, det, img, infer_ms: float, ctx: dict):
        """推論結果をGUIスレッドで適用する（ByteTracker・状態更新・描画）。
        世代が合わない（個体確定後の残留結果）はスキップして None を返す。"""
        if ctx.get('generation') != self._obj_generation:
            return None, YoloResult(self.current_cherry_id, "None", 0.0, cam_name)

        self.obj_infer_ms_sum += infer_ms
        self.obj_infer_count  += 1

        t_post = time.perf_counter()
        tracks = self.trackers[cam_name].update(det, img)

        annotated_frame = img.copy()
        best_result     = YoloResult(ctx['obj_id'], "None", 0.0, cam_name)
        has_valid_track = False
        now_ctx         = ctx.get('now', time.monotonic())

        for row in tracks:
            x1, y1, x2, y2 = map(int, row[:4])
            conf  = float(row[5])
            cls   = int(row[6])
            label = self.model.names[cls].lower()
            if conf < CONF_THRESHOLD:
                continue
            has_valid_track = True
            color      = COLORS.get(label, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            label_text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            back_y1 = max(0, y1 - text_h - 10)
            cv2.rectangle(annotated_frame, (x1, back_y1), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if conf > best_result.confidence:
                best_result = YoloResult(ctx['obj_id'], label, conf, cam_name)

        if has_valid_track:
            self.last_seen_time = now_ctx
            self.obj_last_seen  = now_ctx
            if not self.obj_active:
                self.obj_active     = True
                self.obj_first_seen = now_ctx
                if self.cameras:
                    for cam in self.cameras:
                        cam.reset_cycle_stats()

        self.obj_postproc_ms_sum += (time.perf_counter() - t_post) * 1000.0
        self.obj_postproc_count  += 1

        # 学習用フレーム保持
        if ctx['found'] and ctx['area'] >= MIN_TRAINING_AREA:
            entry = self.obj_cam_train.get(cam_name)
            if entry is None:
                entry = {'min_dist': float('inf'), 'frame': None, 'label': None, 'conf': -1.0}
                self.obj_cam_train[cam_name] = entry
            cd = ctx['center_dist']
            if cd < entry['min_dist']:
                entry['min_dist'] = cd
                entry['frame']    = img.copy()
            if best_result.label_name != "None" and best_result.confidence > entry['conf']:
                entry['conf']  = best_result.confidence
                entry['label'] = best_result.label_name

        # タイル用フレーム + obj_detections への蓄積
        if best_result.label_name != "None":
            if best_result.confidence >= STRICT_THRESHOLDS.get(best_result.label_name, 0.0):
                self.obj_detections.append(best_result)
                self.obj_has_detection = True
            self._update_cam_tile(cam_name, annotated_frame, 2, best_result.confidence)
        elif ctx['found']:
            self._update_cam_tile(cam_name, annotated_frame, 1, -ctx['center_dist'])

        self._buffer_frame(cam_name, annotated_frame)
        return annotated_frame, best_result

    def evaluate_frame(self, frame, cam_name: str) -> tuple:
        now = time.monotonic()

        # ── 1. 前フレームの推論結果を消化（GUIスレッド） ──────────────
        # model.predict はワーカースレッドで完了済み。ByteTracker・状態更新だけここで行う。
        annotated_frame     = None
        best_result_applied = None
        try:
            det, prev_img, infer_ms, ctx = self._result_queues[cam_name].get_nowait()
            annotated_frame, best_result_applied = self._apply_inference(
                cam_name, det, prev_img, infer_ms, ctx)
        except queue.Empty:
            pass

        # ── 2. HSV + in_band 判定 ───────────────────────────────────
        target = ImageProcessor.get_target_info(frame, cam_name)
        found  = target is not None

        in_band = False
        if found:
            w      = frame.shape[1]
            cx     = w / 2.0
            half   = BAND_HALF_PX.get(cam_name, DEFAULT_BAND_HALF_PX)
            xl     = min(max(int(round(cx - half)), 0), w - 1)
            xr     = min(max(int(round(cx + half)), 0), w - 1)
            mi     = target['max_index']
            labs   = target['labels']
            in_band = bool(np.any(labs[:, xl] == mi)) and bool(np.any(labs[:, xr] == mi))

        # ── 3. 出口ヒステリシス ─────────────────────────────────────
        finalized_result = None
        if self.obj_active and (now - self.last_seen_time) >= self.EMPTY_TIMEOUT_SEC:
            finalized_result = self._finalize_object()

        actual_obj_id = self.current_cherry_id

        # ── 4. 表示フレームの決定 ────────────────────────────────────
        # 推論結果がまだ届いていない場合は直近バッファを表示する（1フレーム分の遅延は許容）
        if annotated_frame is None:
            annotated_frame = self.last_frame_per_cam.get(cam_name)
            if annotated_frame is None:
                annotated_frame = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        best_result = best_result_applied or YoloResult(actual_obj_id, "None", 0.0, cam_name)

        # ── 5. 帯外 → 推論投入なし ──────────────────────────────────
        if not in_band:
            raw = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            self._buffer_frame(cam_name, raw)
            return annotated_frame, best_result, finalized_result

        # ── 6. HSVマスク面積比の積算 ─────────────────────────────────
        if found:
            fp = frame.shape[0] * frame.shape[1]
            if fp > 0:
                self.obj_hsv_area_sum   += target['area'] / fp
                self.obj_hsv_area_count += 1

        # ── 7. 前処理（クロップ・リサイズ） ─────────────────────────
        _t_pre = time.perf_counter()
        if found and abs(target['mx'] - frame.shape[1] // 2) < CENTER_THRESHOLD_X:
            input_img   = ImageProcessor.dynamic_crop(frame, target)
            center_dist = abs(target['mx'] - frame.shape[1] // 2)
        else:
            input_img   = frame
            center_dist = frame.shape[1] // 2
        input_img_resized = cv2.resize(input_img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        self.obj_preproc_ms_sum += (time.perf_counter() - _t_pre) * 1000.0
        self.obj_preproc_count  += 1

        # ── 8. model.predict をワーカースレッドへ投入 ────────────────
        # context には状態更新に必要な情報をすべて含める。
        # generation が一致しない古い結果は _apply_inference で破棄される。
        ctx = {
            'generation':  self._obj_generation,
            'obj_id':      actual_obj_id,
            'now':         now,
            'found':       found,
            'center_dist': center_dist,
            'area':        target['area'] if found else 0,
        }
        try:
            self._infer_queue.put_nowait((cam_name, input_img_resized.copy(), ctx))
        except queue.Full:
            pass  # キュー満杯時はこのフレームをドロップして遅延蓄積を防ぐ

        # 今フレームはワーカーが処理中 → 前回の annotated_frame を返して画面を更新し続ける
        return annotated_frame, best_result, finalized_result

    def _buffer_frame(self, cam_name: str, frame) -> None:
        self.last_frame_per_cam[cam_name] = frame

    def close(self) -> None:
        # 推論ワーカースレッドを停止する（None を番兵として送り join する）
        self._infer_running = False
        try:
            self._infer_queue.put_nowait(None)
        except queue.Full:
            pass
        self._infer_thread.join(timeout=2.0)

        # 終了時: 追跡中の個体が残っていれば確定して保存する
        if self.obj_active:
            self._finalize_object()