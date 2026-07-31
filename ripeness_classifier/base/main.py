"""果実熟度分類器（3クラス: 未熟 / 健全 / 過熟）― 本体エントリポイント。

3種類のHSVマスク（果実全体・未熟・過熟）から、未熟マスク・過熟マスクそれぞれの
占有率を直接比較して振り分けるGUI。

判定順（詳細は base/classify.py の classify_3）:
  ①未熟条件（occ_unripe≧未熟占有率境界）・過熟条件（occ_overripe≧過熟占有率境界）の
    両方を満たす場合は、占有率が高い方のクラスを採用する（同点なら未熟を優先）。
  ②未熟条件のみを満たす場合 → unripe（未熟・橙枠）
  ③過熟条件のみを満たす場合 → overripe（過熟・紫枠）
  ④どちらも満たさない場合 → healthy（健全・緑枠）。健全は専用条件を持たない
  ・果実マスクで未検出 → 除外（赤枠）

特徴量の考え方:
  ・未熟マスクのHSV範囲そのものが「未熟の色相」を定義する（黄〜緑の未熟色）。
  ・過熟マスクのHSV範囲そのものが「過熟の色相・明度」を定義する（V上限を絞って暗さを表現）。
  ・つまり色相・明度の個別閾値パラメータは持たず、各マスクのHSV範囲調整＋占有率境界の
    2つの数値だけで3クラスを切り分ける。

重要（間違えやすい点）:
  過熟マスクのV上限は、果実マスクのV下限より高く取ること。そうしないと
  過熟マスク∩果実マスクが常に空になり、過熟占有率が常に0になって過熟を検出できない。

使い方:
  1. 「果実マスク」スライダーで黄〜橙〜赤を包括するHSVを設定（果実全体を拾う）。
  2. 「果柄除去」「虚像除去(彩度)」で果実本体だけを残す（分割された房は自動で1bboxに統合）。
  3. 未熟: 「未熟マスク」スライダーで未熟の色相を調整＋「未熟占有率(%)」で境界を調整。
  4. 過熟: 「過熟マスク」スライダーで過熟の色相・明度を調整＋「過熟占有率(%)」で境界を調整。
  5. 緑=healthy / 橙=unripe / 紫=overripe / 赤=未検出 を確認して設定を保存。
  6. 保存ボタンで画像を3クラスに振り分け出力する。

保存先:
  healthy  : {train}_ripeness_healthy/{class_name}/
  unripe   : {train}_ripeness_unripe/{class_name}/
  overripe : {train}_ripeness_overripe/{class_name}/
"""

import sys
from pathlib import Path

# リポジトリ直下を sys.path に追加し、`python ripeness_classifier/base/main.py` の
# ように直接実行しても `ripeness_classifier.*` を絶対importできるようにする。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtWidgets import QApplication

from ripeness_classifier.base.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
