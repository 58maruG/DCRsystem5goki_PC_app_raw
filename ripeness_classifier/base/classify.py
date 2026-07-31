"""3クラス判定ロジック（未熟マスク・過熟マスク占有率を直接比較する版・表示・集計・保存で共用）。"""


def classify_3(occ_unripe: float, occ_overripe: float, fruit_found: bool,
               occ_unripe_confirm: float, occ_overripe_confirm: float) -> str:
    """未熟マスク占有率・過熟マスク占有率から3クラスを返す。
      occ_unripe           : 未熟占有率 0.0〜1.0。未熟マスク∩果実マスクの面積 ÷ 果実マスク面積
      occ_overripe         : 過熟占有率 0.0〜1.0。過熟マスク∩果実マスクの面積 ÷ 果実マスク面積
      occ_unripe_confirm   : 未熟確定の占有率境界。occ_unripe がこれ以上なら未熟条件を満たす。
      occ_overripe_confirm : 過熟確定の占有率境界。occ_overripe がこれ以上なら過熟条件を満たす。
    判定順:
      ①未熟条件・過熟条件の両方を満たす場合は、occ_unripe と occ_overripe を比較し、
        占有率が高い方のクラスを採用する（同点なら未熟を優先）。
      ②未熟条件のみを満たす場合は未熟。
      ③過熟条件のみを満たす場合は過熟。
      ④どちらも満たさない場合は健全（専用条件を持たない消去法）。
    未熟の色相・明度は未熟マスクのHSV範囲そのもの、過熟の色相・明度は過熟マスクのHSV範囲
    そのもので定義される。分類ロジック自体は各マスクの占有率を境界値と比べるだけで、
    色相・明度の個別閾値パラメータは持たない。"""
    if not fruit_found:
        return "none"
    unripe_ok = occ_unripe >= occ_unripe_confirm
    overripe_ok = occ_overripe >= occ_overripe_confirm
    if unripe_ok and overripe_ok:
        return "unripe" if occ_unripe >= occ_overripe else "overripe"
    if unripe_ok:
        return "unripe"
    if overripe_ok:
        return "overripe"
    return "healthy"
