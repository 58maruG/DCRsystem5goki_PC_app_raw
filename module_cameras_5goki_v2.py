from __future__ import annotations
import os
import time
import threading
import gc
from pypylon import pylon
from collections import deque
from PySide6.QtCore import QObject, Signal

import log_config
log = log_config.get_logger("camera")

# ================================================
# カメラ設定
# ================================================
TARGET_SERIALS = [
    ("25453227", "cam_top"),
    ("25453229", "cam_under"),
    ("25308967", "cam_inside"),
    ("25308968", "cam_outside"),
]
FPS = 20.0


# ================================================
# PFSファイル読み込みヘルパー
# ================================================
def load_pfs_custom(camera, pfs_path: str) -> bool:
    if not os.path.exists(pfs_path):
        return False
    try:
        pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap(), True)
        return True
    except Exception as e:
        log.error("PFSロードエラー (%s): %s", pfs_path, e)
        return False


# ================================================
# シグナル送信用ヘルパークラス
# ================================================
class CameraSignals(QObject):
    connection_lost = Signal(str)


# ================================================
# カメラ制御クラス（映像取得・表示専用版）
# ================================================
class CameraController(QObject):
    def __init__(self, device_info, cam_name: str = "unknown") -> None:
        super().__init__()

        # --- シグナル ---
        self.signals = CameraSignals()

        # --- デバイス情報 ---
        self.device_info   = device_info
        self.name          = cam_name
        self.serial        = device_info.GetSerialNumber()
        self.settings_file = f"cam_pfs/{self.name}_{self.serial}.pfs"

        # --- カメラオブジェクト・フレーム管理 ---
        self.camera        = None
        self.is_capturing  = False
        self.thread        = None
        self.latest_frame  = None
        self.lock          = threading.Lock()
        self.frame_queue   = deque()
        self.delay_seconds = 0.0

        # --- サイクル統計（reset_cycle_stats / get_cycle_stats でアクセス） ---
        self._cycle_lock              = threading.Lock()
        self._cycle_drop_count        = 0
        self._last_capture_latency_ms = None

        # --- Pylon 画像変換器 ---
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # --- 解像度（init_camera 後に実値で上書き） ---
        self.width  = 1280
        self.height = 960

    # --- カメラ初期化 ---
    def init_camera(self) -> bool:
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateDevice(self.device_info)
            )
            self.camera.Open()
            self.camera.MaxNumBuffer = 20

            # 各カメラが最大 100MB/s 以上出さないように制限（4台で400MB/s = USB3.0の実効限界付近）
            if hasattr(self.camera, 'DeviceLinkThroughputLimit'):
                self.camera.DeviceLinkThroughputLimitMode.Value = "On"
                self.camera.DeviceLinkThroughputLimit.Value = 100000000  # 100MB/s

            if os.path.exists(self.settings_file):
                success = load_pfs_custom(self.camera, self.settings_file)
                if success:
                    log.info("%s: 設定を適用しました", self.name)
                else:
                    log.warning("%s: 設定解析失敗", self.name)

            self.width  = self.camera.Width.Value
            self.height = self.camera.Height.Value
            return True
        except Exception as e:
            log.error("カメラ初期化エラー (%s): %s", self.name, e)
            return False

    # --- キャプチャ開始 ---
    def start_capture(self) -> None:
        if not self.camera or not self.camera.IsOpen():
            return
        self.is_capturing = True
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        log.info("キャプチャ開始: %s", self.name)

    # --- キャプチャループ（内部スレッド） ---
    def _capture_loop(self) -> None:
        # --- 再発頻度・原因特定用の統計 ---
        consecutive_errors = 0          # 現在の連続エラー回数
        error_streak_start = None       # 連続エラーが始まった時刻 (monotonic)
        total_errors       = 0          # このセッションの累積エラー回数
        total_frames       = 0          # このセッションの累積取得成功フレーム数
        last_stats_log     = time.monotonic()
        FATAL_TIMEOUT_SEC  = 5.0        # この秒数だけ連続で1枚も取得できなければ接続喪失とみなす
        STATS_INTERVAL_SEC = 60.0       # 定期ヘルスログの間隔

        while self.is_capturing:
            # カメラオブジェクト自体が失われている場合のチェック
            if not self.camera:
                self.signals.connection_lost.emit(self.name)
                break

            try:
                # タイムアウトを5秒に設定して画像を取得（取得時間を計測）
                _t_grab    = time.perf_counter()
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                _capture_ms = (time.perf_counter() - _t_grab) * 1000.0
                try:
                    if grab_result.GrabSucceeded():
                        # --- 取得成功 --- レイテンシを記録
                        with self._cycle_lock:
                            self._last_capture_latency_ms = round(_capture_ms, 2)
                        # 直前まで連続エラーだった場合は、フリーズしていた時間をログに残す
                        if consecutive_errors > 0:
                            frozen_ms = (time.monotonic() - error_streak_start) * 1000.0 \
                                        if error_streak_start else 0.0
                            log.warning("%s: %d回の連続エラーから復帰 (この間パネルは約 %.0fms フリーズ)",
                                        self.name, consecutive_errors, frozen_ms)
                        consecutive_errors = 0
                        error_streak_start = None
                        total_frames += 1

                        converted  = self.converter.Convert(grab_result)
                        frame_bgr  = converted.GetArray()

                        with self.lock:
                            delay_frames = int(self.delay_seconds * FPS)
                            if delay_frames > 0:
                                self.frame_queue.append(frame_bgr.copy())
                                if len(self.frame_queue) > delay_frames:
                                    self.latest_frame = self.frame_queue.popleft()
                                else:
                                    self.latest_frame = None
                            else:
                                self.latest_frame = frame_bgr.copy()
                                self.frame_queue.clear()
                    else:
                        # 取得失敗の理由(Pylonエラーコード)を残して例外へ
                        raise Exception(
                            f"GrabSucceeded=False code=0x{grab_result.GetErrorCode():08X} "
                            f"desc={grab_result.GetErrorDescription()}"
                        )
                finally:
                    # 成否に関わらずバッファを必ず解放（リーク防止）
                    grab_result.Release()

                # --- 定期ヘルスログ（再発頻度の長期モニタ用） ---
                now = time.monotonic()
                if now - last_stats_log >= STATS_INTERVAL_SEC:
                    denom    = total_frames + total_errors
                    err_rate = (total_errors / denom * 100.0) if denom else 0.0
                    msg = ("%s: 累積 成功 %d 枚 / エラー %d 回 (エラー率 %.2f%%)"
                           % (self.name, total_frames, total_errors, err_rate))
                    # エラーが起きたセッションは WARN（コンソール+ファイル）、正常時は DEBUG（ファイルのみ）
                    if total_errors > 0:
                        log.warning("[CamStats] %s", msg)
                    else:
                        log.debug("[CamStats] %s", msg)
                    last_stats_log = now

            except Exception as e:
                # --- 取得失敗（タイムアウト等）。スリープせず即座に再試行する ---
                with self._cycle_lock:
                    self._cycle_drop_count += 1
                total_errors += 1
                consecutive_errors += 1
                now = time.monotonic()
                if error_streak_start is None:
                    error_streak_start = now
                elapsed = now - error_streak_start

                # 原因特定用の詳細ログ（カメラ名・連続回数・継続時間・累積・例外種別/コード）
                log.warning("[GrabError] %s: 連続 %d 回目 / 継続 %.0fms / 累積エラー %d 回・成功 %d 枚 - %s: %s",
                            self.name, consecutive_errors, elapsed * 1000,
                            total_errors, total_frames, type(e).__name__, e)

                # 一定時間ずっと取得できないときだけ致命的とみなして通知（瞬断では止めない）
                if elapsed >= FATAL_TIMEOUT_SEC:
                    log.error("[Fatal] %s: %.1f 秒間フレーム取得不能。接続喪失として通知します。",
                              self.name, elapsed)
                    self.signals.connection_lost.emit(self.name)
                    break

                # スリープせず即再試行。グラビングが止まっていた場合のみ再開する。
                try:
                    if not self.camera.IsGrabbing():
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                except Exception as re:
                    log.error("[Recover] %s: StartGrabbing 再開に失敗 - %s: %s",
                              self.name, type(re).__name__, re)

        log.debug("スレッド終了: %s", self.name)

    # --- キャプチャ停止 ---
    def stop_capture(self) -> None:
        self.is_capturing = False
        # StopGrabbingを先に呼んでRetrieveResultをキャンセルさせてからjoinする
        # （逆順だとjoinがRetrieveResultの5秒タイムアウトを待ってしまう）
        if self.camera and self.camera.IsGrabbing():
            try:
                self.camera.StopGrabbing()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=3.0)

    # --- サイクル統計リセット（新個体の検出開始時） ---
    def reset_cycle_stats(self) -> None:
        with self._cycle_lock:
            self._cycle_drop_count = 0

    # --- サイクル統計取得 ---
    def get_cycle_stats(self) -> dict:
        with self._cycle_lock:
            return {
                'capture_latency_ms': self._last_capture_latency_ms,
                'frame_dropped':      self._cycle_drop_count,
            }

    # --- 最新フレーム取得 ---
    def get_current_frame(self):
        with self.lock:
            return self.latest_frame

    # --- リソース完全解放 ---
    def close(self) -> None:
        self.stop_capture()
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
            except Exception as e:
                log.error("カメラ解放エラー (%s): %s", self.name, e)
            finally:
                # InstantCamera デストラクタがデバイスを解放するため、参照を手放すだけでよい
                self.camera = None
        log.info("%s: リソース解放完了", self.name)


# ================================================
# カメラマネージャ
# ================================================
class CameraManager:
    def __init__(self) -> None:
        self.controllers: list[CameraController] = []

    # --- 全カメラ初期化 ---
    def init_cameras(self) -> bool:
        # 1. 既存のコントローラーを完全に破棄
        for controller in self.controllers:
            controller.close()
        self.controllers.clear()

        # 2. Pythonのガベージコレクションを強制実行してOSにリソースを戻す
        gc.collect()
        time.sleep(0.5)  # ドライバが専有権を戻すための物理的な猶予

        try:
            tl_factory = pylon.TlFactory.GetInstance()
            # キャッシュをクリア（これが重要！）
            devices = tl_factory.EnumerateDevices()
        except Exception as e:
            log.error("Pylon初期化エラー: %s", e)
            return False

        if not devices:
            log.error("カメラが見つかりません")
            return False

        # 3. 指定したシリアルに一致するカメラのみを開く
        for target_serial, cam_name in TARGET_SERIALS:
            found = next((d for d in devices if d.GetSerialNumber() == target_serial), None)
            if found:
                controller = CameraController(found, cam_name)
                if controller.init_camera():
                    self.controllers.append(controller)
                else:
                    # 初期化に失敗した場合はリソースを即座に捨てる
                    controller.close()
                    del controller

        return len(self.controllers) > 0

    # --- 全カメラキャプチャ開始 ---
    def start_all_get_frame(self) -> None:
        for controller in self.controllers:
            controller.start_capture()

    # --- 全カメラキャプチャ停止・解放 ---
    def stop_all_get_frame(self) -> None:
        for controller in self.controllers:
            controller.close()
