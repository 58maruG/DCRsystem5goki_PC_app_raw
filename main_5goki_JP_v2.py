# -------------------------------------------------
# main_5goki_JP_v2.py
#   黒タイル対策の統合設計版エントリポイント。判定ロジックを
#   module_yolo_csv4（状態機械＋フレーム保持一本化）に差し替えただけで、
#   その他は main_5goki_JP.py と同一。旧版はそのまま温存してある。
# -------------------------------------------------
import sys
import requests
import cv2
import time

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Slot, Qt, QRunnable, QThreadPool, QTimer
from PySide6.QtGui import QKeyEvent, QImage, QPixmap

# GUIモジュール
import module_gui_JP

# 制御モジュール
import module_cameras_5goki_v2 as cam_ctr
import module_relay as r_ctr
import module_patlite as p_ctr
import module_yolo_csv4 as yolo_ctr

RPI_IP_ADDRESS = "192.168.2.1"
RPI_PORT = 5000

# リレーモジュール側の設定と同期させるための定数
SPEED_MAP = {
    1: 0.0010, 2: 0.0009, 3: 0.0008, 4: 0.0007,
    5: 0.0006, # 基準 (デフォルト)
    6: 0.0005, 7: 0.0004, 8: 0.0003, 9: 0.0002, 10: 0.0001
}
# ステッピングモータの設定変数
RATIO = 1.0                 # 基本補正係数
MICRO_STATUS = 32           # マイクロステップ設定

# ==========================================================
# 汎用バックグラウンドタスク用クラス
# ==========================================================
class TaskWorker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            print(f"!![Background Task Error]: {e}")

# ==========================================================
# スタートアップウィンドウ
# ==========================================================
class StartupWindow(module_gui_JP.StartupWindowUI):
    def __init__(self):
        super().__init__()
        self.button_start.clicked.connect(self.launch_main)

    def launch_main(self):
        self.main_window = MainWindow()
        self.main_window.showFullScreen()
        self.close()

# ==========================================================
# サブウィンドウ
# ==========================================================
class SubWindow(module_gui_JP.SubWindowUI):
    def __init__(self, parent_window, initial_speed):
        super().__init__()
        self.button_up_speed.clicked.connect(self.on_up_speed)
        self.button_down_speed.clicked.connect(self.on_down_speed)
        self.button_back.clicked.connect(self.go_back)

        self.parent_window = parent_window
        self.current_speed = initial_speed
        self.update_speed_ui()

    def update_speed_ui(self):
        self.label_current_speed.setText(str(self.current_speed))

        if self.current_speed >= 10:
            self.button_up_speed.set_locked(True)
        else:
            self.button_up_speed.set_locked(False)

        if self.current_speed <= 1:
            self.button_down_speed.set_locked(True)
        else:
            self.button_down_speed.set_locked(False)

    @Slot()
    def on_up_speed(self):
        if self.current_speed < 10:
            self.current_speed += 1
            self.update_speed_ui()

    @Slot()
    def on_down_speed(self):
        if self.current_speed > 1:
            self.current_speed -= 1
            self.update_speed_ui()

    @Slot()
    def go_back(self):
        self.parent_window.saved_speed = self.current_speed
        self.close()

# ==========================================================
# カメラエラーウィンドウ
# ==========================================================
class CameraErrorWindow(module_gui_JP.CameraErrorWindowUI):
    def __init__(self, parent_window, lost_cam_name):
        super().__init__(lost_cam_name)
        self.parent_window = parent_window
        self.button_continue.clicked.connect(self.attempt_recovery)

    def attempt_recovery(self):
        print("\n>>> 復旧プロセス開始...")
        self.button_continue.setEnabled(False)
        self.label_cams.setText("全接続を切断中...")
        QApplication.processEvents()

        self.parent_window.cameras.stop_all_get_frame()
        time.sleep(1.0)

        if self.parent_window.cameras.init_cameras():
            connected_names = [c.name for c in self.parent_window.cameras.controllers]
            required_names = [name for _, name in cam_ctr.TARGET_SERIALS]
            missing = set(required_names) - set(connected_names)

            if not missing and len(connected_names) == 4:
                print(">>> 4台全てのカメラが正常に再オープンされました。")

                for controller in self.parent_window.cameras.controllers:
                    controller.signals.connection_lost.connect(self.parent_window.handle_camera_error)

                self.parent_window.cameras.start_all_get_frame()
                self.parent_window.run_in_background(self.parent_window.relay.stop)

                # 復旧完了 → 待機中状態に戻す (ブザー停止、赤点灯)
                self.parent_window.run_in_background(
                    self.parent_window.patlite.set_system_state, p_ctr.SystemState.STANDBY
                )

                self.parent_window.timer.start(50)
                self.close()
            else:
                msg = f"不完全: {len(connected_names)}/4 台のカメラ\n"
                if missing:
                    msg += f"未接続: {', '.join(missing)}"
                self.label_cams.setText(msg)
                self.button_continue.setEnabled(True)
        else:
            self.label_cams.setText("カメラを開けませんでした。\nUSBケーブルを確認してください。")
            self.button_continue.setEnabled(True)

# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(module_gui_JP.MainWindowUI):
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()

        self.history_data = []
        self.current_id = 1

        # --- デバイス接続 (接続中はYELLOW) ---
        self.patlite = p_ctr.PatliteController()
        if not self.patlite.init():
            print("!!パトライトの接続に失敗しました")
            self.close()

        # 初期化中を表示
        self.patlite.set_system_state(p_ctr.SystemState.INITIALIZING)

        self.relay = r_ctr.RelayController()
        if not self.relay.init():
            print("!!リレーボードの接続に失敗しました")
            self.close()

        self.cameras = cam_ctr.CameraManager()
        if not self.cameras.init_cameras():
            print("!!カメラの接続に失敗しました")
            self.close()

        # カメラエラーハンドラの設定
        for controller in self.cameras.controllers:
            controller.signals.connection_lost.connect(self.handle_camera_error)

        # カウント用辞書の初期化
        self.detection_counts = {
            "healthy": 0, "twin": 0, "unripe": 0, "mold": 0, "stemcrack": 0, "birddamage": 0,
            "malformation": 0, "crack": 0, "wilt": 0, "suturecrack": 0, "brownrot": 0, "blacktwin": 0,
            "insect": 0
        }
        self.update_stats_display()

        # YOLO初期化（モデルパスは module_yolo_csv3.py の MODEL_PATH で一元管理）
        self.detector = yolo_ctr.YoloDetector()

        # スピード初期値設定とカメラ遅延の初期適用
        self.saved_speed = 5
        self.update_camera_delays(self.saved_speed)

        self.cameras.start_all_get_frame()

        # イベント接続
        self.toggle_switch.toggled.connect(self.on_main_toggled)
        self.button_setting.clicked.connect(self.on_setting_button)
        self.button_power.clicked.connect(self.on_power_bottom)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feeds)
        self.timer.start(50)

        # 全デバイス接続完了 → 待機中 (RED、ブザーなし)
        self.patlite.set_system_state(p_ctr.SystemState.STANDBY)

    # --- カメラ表示遅延をスピードに合わせて更新する関数 ---
    def update_camera_delays(self, speed):
        delay = SPEED_MAP[speed]
        t_one_pulse = delay * 2
        step_one_rotation = RATIO * (360 / 1.8) * MICRO_STATUS
        sec = t_one_pulse * step_one_rotation * 2   # ギア比が2なので

        dynamic_delay = sec * (60 / 360)
        DYNAMIC_UNDER_DELAY = 2.367
        DYNAMIC_INSIDE_DELAY = 2.594

        print(f"[Sync] Speed:{speed} に基づき表示遅延を {dynamic_delay:.2f} 秒に更新します")

        for controller in self.cameras.controllers:
            if controller.name == "cam_under":
                controller.delay_seconds = DYNAMIC_UNDER_DELAY
            elif controller.name == "cam_inside":
                controller.delay_seconds = DYNAMIC_INSIDE_DELAY

    # --- カメラ映像をGUIに反映する関数 ---
    def update_video_feeds(self):
        is_running = self.toggle_switch.isChecked()

        for controller in self.cameras.controllers:
            frame = controller.get_current_frame()
            if frame is not None:
                if is_running:
                    annotated_frame, result, finalized_result = self.detector.evaluate_frame(
                        frame, controller.name, self.current_id
                    )
                    if finalized_result is not None:
                        self.process_final_result(finalized_result)
                    display_frame = annotated_frame
                else:
                    display_frame = frame

                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                target_label = None
                if controller.name == "cam_inside":
                    target_label = self.cam_in
                elif controller.name == "cam_outside":
                    target_label = self.cam_out
                elif controller.name == "cam_under":
                    target_label = self.cam_under
                elif controller.name == "cam_top":
                    target_label = self.cam_top

                if target_label:
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(
                        target_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    target_label.setPixmap(scaled_pixmap)

    # --- バックグラウンドで渡された関数を実行するヘルパー関数 ---
    def run_in_background(self, func, *args, **kwargs):
        worker = TaskWorker(func, *args, **kwargs)
        self.thread_pool.start(worker)

    # --- ラズパイと通信する関数 ---
    def __async_raspi_request(self, command):
        url = f"http://{RPI_IP_ADDRESS}:{RPI_PORT}{command}"
        try:
            print(f">>>[Sending]: {url}")
            requests.get(url, timeout=2)
        except Exception as e:
            print(f"!![Net Error]: {e}")

    # --- 設定ボタン押下イベント ---
    @Slot()
    def on_setting_button(self):
        self.settings_window = SubWindow(parent_window=self, initial_speed=self.saved_speed)
        self.settings_window.show()

    # --- カメラエラー発生時の処理 ---
    @Slot(str)
    def handle_camera_error(self, cam_name):
        print(f"!![Emergency Stop] カメラ '{cam_name}' の接続が切れました。")

        # GUIのトグルをOFFにする
        self.toggle_switch.setChecked(False)

        # タイマー（推論と表示更新）を停止
        self.timer.stop()

        # 他のカメラの取得も完全に止める
        self.cameras.stop_all_get_frame()

        # エラー状態: RED + 継続ブザー
        self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.ERROR)

        # エラーウィンドウのポップアップ
        self.error_win = CameraErrorWindow(self, cam_name)
        self.error_win.show()

    # --- 電源ボタン押下イベント ---
    @Slot()
    def on_power_bottom(self):
        print("\n電源ボタンが押されました。終了します。\n")
        self.timer.stop()

        self.patlite.close()    # 消灯してから切断
        self.relay.close()
        self.cameras.stop_all_get_frame()
        self.run_in_background(self.__async_raspi_request, "/cleanup_system")
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()

        self.close()

    # --- 確定した推論結果をGUIとリレーに反映する関数 ---
    def process_final_result(self, result_obj):
        if not self.toggle_switch.isChecked():
            return

        disease_name = result_obj.label_name
        confidence_percent = int(result_obj.confidence * 100)
        obj_id = result_obj.id

        channel = None
        display_name = ""

        if disease_name in self.detection_counts:
            self.detection_counts[disease_name] += 1
            self.update_stats_display()

        if disease_name == "healthy":
            channel = r_ctr.RelayChannel.TRANSPORT
            display_name = "健全果"
            self.label_dam.setText("健全果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #FFFFFF;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "twin":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "双子果"
            self.label_dam.setText("双子果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #FF0000;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "unripe":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "未熟果"
            self.label_dam.setText("未熟果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #FFFF00;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "mold":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "カビ"
            self.label_dam.setText("カビ")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #800080;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "stemcrack":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "果梗裂果"
            self.label_dam.setText("果梗裂果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #0000FF;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "birddamage":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "鳥害"
            self.label_dam.setText("鳥害")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #4169E1;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "malformation":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "奇形果"
            self.label_dam.setText("奇形果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #FF4500;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "crack":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "裂果"
            self.label_dam.setText("裂果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #00BFFF;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "wilt":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "萎凋果"
            self.label_dam.setText("萎凋果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #8B4513;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "suturecrack":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "縫合線裂果"
            self.label_dam.setText("縫合線裂果")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #000000; background-color: #20B2AA;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "brownrot":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "灰星病"
            self.label_dam.setText("灰星病")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #A0522D;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "blacktwin":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "黒双子"
            self.label_dam.setText("黒双子")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #2F4F4F;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)
        elif disease_name == "insect":
            channel = r_ctr.RelayChannel.REMOVE
            display_name = "虫害"
            self.label_dam.setText("虫害")
            self.label_dam.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #FFFFFF; background-color: #556B2F;
                border: 1px solid #000000;
                qproperty-alignment: 'AlignCenter';
            """)

        if channel is not None:
            # リレー制御のみ (パトライトはシステム状態で常時GREEN固定)
            self.run_in_background(self.relay.move, channel, self.saved_speed)

            record = {
                "id": obj_id,
                "result": display_name,
                "conf": confidence_percent
            }
            self.history_data.append(record)
            if len(self.history_data) > 10:
                self.history_data.pop(0)
            print(f"Latest History: | ID: {record['id']:03} | 判定結果: {record['result']} | 信頼度: {record['conf']} % |")
            self.update_history_display()

    # --- 統計情報を更新する関数 ---
    def update_stats_display(self):
        if hasattr(self, 'label_stats'):
            self.label_stats.setText("入力待機中...")

    # --- 履歴表示を更新する関数 (HTMLテーブル版) ---
    def update_history_display(self):
        rows_html = ""
        for item in self.history_data:
            id_txt = f"{item['id']:03}".translate(str.maketrans("0123456789", "０１２３４５６７８９"))
            raw_text = item['result']

            if "健全果" in raw_text:
                color_code = "#ffffff"
            elif "黒双子" in raw_text:
                color_code = "#2F4F4F"
            elif "双子果" in raw_text:
                color_code = "#FF0000"
            elif "未熟果" in raw_text:
                color_code = "#FFFF00"
            elif "カビ" in raw_text:
                color_code = "#800080"
            elif "果梗裂果" in raw_text:
                color_code = "#0000FF"
            elif "鳥害" in raw_text:
                color_code = "#4169E1"
            elif "奇形果" in raw_text:
                color_code = "#FF4500"
            elif "縫合線裂果" in raw_text:
                color_code = "#20B2AA"
            elif "裂果" in raw_text:
                color_code = "#00BFFF"
            elif "萎凋果" in raw_text:
                color_code = "#8B4513"
            elif "灰星病" in raw_text:
                color_code = "#A0522D"
            else:
                color_code = "#00FF00"

            conf_txt = f"{item['conf']} ％".translate(str.maketrans("0123456789", "０１２３４５６７８９"))

            rows_html += f"""
            <tr>
                <td align="center" style="border-right: 1px solid #00FF00;">{id_txt}</td>
                <td align="center" style="border-right: 1px solid #00FF00; color:{color_code};">{raw_text}</td>
                <td align="center" style="border-right: 1px solid #00FF00;">{conf_txt}</td>
            </tr>
            """

        c = self.detection_counts
        stats_html = f"""
        <html>
        <body style="background-color:#000000; color:#00FF00; font-family:'MS Gothic';">
            <table width="100%" style="border: none;">
                <tr>
                    <td>健全果: {c['healthy']}</td>
                    <td>双子果: {c['twin']}</td>
                </tr>
                <tr>
                    <td>未熟果: {c['unripe']}</td>
                    <td>カビ: {c['mold']}</td>
                </tr>
                <tr>
                    <td>果梗裂果: {c['stemcrack']}</td>
                    <td>鳥害: {c['birddamage']}</td>
                </tr>
                <tr>
                    <td>奇形果: {c['malformation']}</td>
                    <td>裂果: {c['crack']}</td>
                </tr>
                <tr>
                    <td>萎凋果: {c['wilt']}</td>
                    <td>縫合線裂果: {c['suturecrack']}</td>
                </tr>
                <tr>
                    <td>灰星病: {c['brownrot']}</td>
                    <td>黒双子: {c['blacktwin']}</td>
                </tr>
                <tr>
                    <td>虫害: {c['insect']}</td>
                    <td></td>
                </tr>
            </table>
        </body>
        </html>
        """
        self.label_stats.setText(stats_html)

        full_html = f"""
        <html>
        <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                border: 1px solid #00FF00;
            }}
            th {{
                font-family: "MS Gothic"; font-size: 20px; font-weight: bold; color: #00FF00;
                border-right: 1px solid #00FF00;
                padding: 4px;
            }}
            td {{
                font-family: "MS Gothic"; font-size: 16px; font-weight: bold; color: #00FF00;
                padding: 3px;
            }}
        </style>
        </head>
        <body style="background-color:#000000;">
            <table cellspacing="0">
                <tr>
                    <th width="20%">ＩＤ</th>
                    <th width="40%">結果</th>
                    <th width="40%">信頼度</th>
                </tr>
                {rows_html}
            </table>
        </body>
        </html>
        """
        self.label_history.setText(full_html)

    # --- トグルスイッチ状態変更イベント ---
    @Slot(bool)
    def on_main_toggled(self, checked):
        self.button_setting.set_locked(checked)
        if checked:
            self.update_camera_delays(self.saved_speed)
            self.label_toggle_status.setText("動作中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #32CD32; qproperty-alignment: 'AlignCenter';
            """)
            self.run_in_background(self.__async_raspi_request, f"/set_speed/{self.saved_speed}")
            print(f"\nスピード設定をメインに保存: {self.saved_speed}")
            self.run_in_background(self.__async_raspi_request, "/rotate")
            # 正常運転中: GREEN、ブザーなし
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.RUNNING)
        else:
            self.label_toggle_status.setText("停止中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #888888; qproperty-alignment: 'AlignCenter';
            """)
            self.run_in_background(self.__async_raspi_request, "/stop")
            self.run_in_background(self.relay.stop)
            # 手動停止・待機中: RED、ブザーなし
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDBY)

# ==========================================================
# 実行ブロック
# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartupWindow()
    window.show()
    sys.exit(app.exec())
