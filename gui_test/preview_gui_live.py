# -------------------------------------------------
# GUIライブプレビュー用スタンドアロンスクリプト
#   カメラ・デバイスなしで、1秒ごとにランダムな検出結果を生成し、
#   検出履歴リストと個数スタック欄をリアルタイムに更新して見た目を確認する。
#   HTML組み立ては preview_gui.py の関数を再利用する。
#   実行: python preview_gui_live.py
# -------------------------------------------------
import os
import sys
import random

# このスクリプトを gui_test/ など子フォルダに置いても、親フォルダの
# module_gui_JP_v3 を import できるよう、親フォルダをパスに追加する。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

import module_gui_JP_v3 as gui
from preview_gui import build_history_html, build_stats_html, build_status_html

# 生成間隔（ミリ秒）
TICK_MS = 1000
# 履歴リストに残す最大件数（main_5goki_JP_v3.py と同じく5件）
HISTORY_MAX = 5

# strict閾値（本番 module_yolo_csv4_v3.py の STRICT_THRESHOLDS と同値をミラー）。
#   ここに無いクラスは閾値なし＝常に表示。
STRICT_THRESHOLDS = {
    "stemcrack": 0.85, "crack": 0.85, "birddamage": 0.85,
    "twin": 0.9, "blacktwin": 0.9, "malformation": 0.9, "unripe": 0.9,
}


def random_detection(detection_id: int) -> dict:
    """ランダムな1件の検出結果を生成する。本番 _resolve_best_result を模擬する。
    閾値ありクラスは閾値未満を無効化するが、有効な検出が無ければ検出信頼度が最大の
    クラスで確定する（閾値未満でも確定）。表示順は「確定クラスを左端 → 残りは信頼度降順」。"""
    classes = list(gui.CLASS_DISPLAY.keys())
    n = random.randint(1, 4)                       # 検出クラス数（1〜4）
    chosen = random.sample(classes, n)             # 重複なしでn個選ぶ
    confs = sorted((random.uniform(0.50, 0.99) for _ in range(n)), reverse=True)
    raw = list(zip(chosen, confs))

    # strict閾値未満のクラスは無効化（無かった扱い）。閾値なしクラスは常に有効。
    eligible = [(lbl, conf) for lbl, conf in raw if conf >= STRICT_THRESHOLDS.get(lbl, 0.0)]
    if not eligible:
        # 有効な検出なし → 検出信頼度が最大のクラスで確定（確定クラスのみ表示）
        final, conf = max(raw, key=lambda x: x[1])
        return {"id": detection_id, "final": final, "breakdown": [(final, conf)]}

    final = random.choice([lbl for lbl, _ in eligible])   # 確定クラス（最大信頼度とは限らない）
    confirmed = [bd for bd in eligible if bd[0] == final]
    others    = sorted((bd for bd in eligible if bd[0] != final),
                       key=lambda bd: bd[1], reverse=True)
    return {
        "id": detection_id,
        "final": final,
        "breakdown": confirmed + others,           # 確定クラスを左端へ
    }


class LiveSimulator:
    """1秒ごとにランダム検出を生成し、GUIへ反映するシミュレータ。"""

    def __init__(self, window: gui.MainWindowUI) -> None:
        self.window = window
        self.history: list[dict] = []
        self.counts = {label: 0 for label in gui.CLASS_DISPLAY}
        self.next_id = 1

        self.timer = QTimer(window)
        self.timer.timeout.connect(self.tick)

    def start(self) -> None:
        self.timer.start(TICK_MS)

    def tick(self) -> None:
        det = random_detection(self.next_id)
        self.next_id += 1
        self.counts[det["final"]] += 1

        self.history.append(det)
        if len(self.history) > HISTORY_MAX:
            self.history.pop(0)

        self.window.label_history.setText(build_history_html(self.history))
        self.window.label_stats.setText(build_stats_html(self.counts))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = gui.MainWindowUI()

    # 初期表示（空のリスト＋カウント0）
    window.label_history.setText(build_history_html([]))
    window.label_stats.setText(build_stats_html({label: 0 for label in gui.CLASS_DISPLAY}))

    # ステータスバー・モード表示パネル（動作中の見た目で固定）
    window.label_status.setText(build_status_html("run"))
    window.label_mode.setText("PCモード")
    window.label_mode.setStyleSheet(gui.LABEL_MODE_STYLE_PC)
    window.label_pulse_speed.setText("6")
    window.label_model.setText("best_v3.pt")

    simulator = LiveSimulator(window)
    simulator.start()

    window.show()
    sys.exit(app.exec())
