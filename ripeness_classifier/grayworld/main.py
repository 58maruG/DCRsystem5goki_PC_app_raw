"""果実熟度分類器（3クラス: 未熟 / 健全 / 過熟）【グレーワールド正規化版】エントリポイント。

カメラごとの色温度差を吸収するため、各画像に per-image グレーワールド正規化
（各チャンネル平均を揃えて白色点を中性化）を掛けてから判定・表示する。
  ・正規化は「判定＋サムネイル表示」にのみ適用（HSV調整が直感的になる）。
  ・保存画像は元の色のまま（コピー=元ファイル、クロップ=元画像から切り出し）。
  ・設定JSONは本体とは別ファイル（hsv_ripeness_config_grayworld.json）。
  ・GUIの「グレーワールド正規化」チェックでON/OFF比較可能（既定ON）。
注意: グレーワールドは画像全体の平均を使うため、背景が多色（内/外カメラのアクリル）だと
      補正が偏ることがある。背景がほぼ白（上/下カメラ）では安定しやすい。

分類ロジックは本体(base)に追従する（base/classify.py の classify_3）:
  ①未熟条件（occ_unripe≧未熟占有率境界）・過熟条件（occ_overripe≧過熟占有率境界）の
    両方を満たす場合は、占有率が高い方のクラスを採用する（同点なら未熟を優先）。
  ②未熟条件のみを満たす場合 → unripe（未熟・橙枠）
  ③過熟条件のみを満たす場合 → overripe（過熟・紫枠）
  ④どちらも満たさない場合 → healthy（健全・緑枠）
  ・果実マスクで未検出 → 除外（赤枠）
grayworld 版はこの判定の前にグレーワールド正規化を挟むだけで、判定式そのものは base と同一。

保存先:
  healthy  : {train}_ripeness_healthy/{class_name}/
  unripe   : {train}_ripeness_unripe/{class_name}/
  overripe : {train}_ripeness_overripe/{class_name}/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtWidgets import QApplication

from ripeness_classifier.grayworld.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
