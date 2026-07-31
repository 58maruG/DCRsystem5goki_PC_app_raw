"""3クラス判定ロジック（グレーワールド版）。

本体(base)の最新ロジック（未熟マスク占有率・過熟マスク占有率の直接比較）へ追従するため、
判定そのものは base.classify.classify_3 をそのまま再利用する。grayworld 版に固有なのは
「判定前にグレーワールド正規化を掛けるかどうか」という前処理のみ（processing.py 側で対応）。
"""

from ripeness_classifier.base.classify import classify_3

__all__ = ["classify_3"]
