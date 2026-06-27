# -*- coding: utf-8 -*-
"""
cycle ログを cycle_id 昇順に並べ替えて書き出すユーティリティ（DuckDB）。

背景:
  cycle ログは「確定スナップショット行」（capture/infer/preproc 等が埋まる行）と
  「排出結果行」（planned_eject_ts/eject_delay/outcome_flag のみの行）が時間差で
  別々に追記される。排出はバックグラウンドで物理動作の完了を待ってから書かれるため、
  ファイル上では cycle_id が前後して見える（timestamp 順では正しく昇順）。

  本スクリプトは記録済みデータを並べ替えるだけで、元ファイルは変更しない。

出力:
  - <name>.csv          : 全行を cycle_id 昇順（同 id 内は timestamp 昇順）に並べ替え
  - <name>_merged.csv   : 1 サイクル 1 行に統合（確定行＋排出結果行をマージ）

使い方:
  python analysis/sort_cycle.py                   # logs_cycle_5goki/*.csv を一括処理
  python analysis/sort_cycle.py path/to/x.csv ... # 指定ファイルのみ
出力先:
  analysis/sorted/
"""

from __future__ import annotations

import os
import sys
import glob

try:
    import duckdb
except ImportError:
    print("duckdb が見つかりません。`pip install duckdb` を実行してください。")
    sys.exit(1)

# Windows コンソールの文字化け（℃ など）対策
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "analysis", "sorted")

# 排出結果行だけが持つ列（確定行では空）。NULL 集約で 1 行に畳むときの対象。
EJECT_COLS = ["planned_eject_ts", "eject_delay[ms]", "outcome_flag"]


def to_uri(path: str) -> str:
    """DuckDB の文字列リテラル用にパスを整形（バックスラッシュ→スラッシュ）。"""
    return path.replace("\\", "/")


def read_view(con, src: str) -> None:
    """対象 CSV を全列 VARCHAR のビュー src_view として読み込む。"""
    uri = to_uri(src)
    con.execute(
        "CREATE OR REPLACE VIEW src_view AS "
        f"SELECT * FROM read_csv('{uri}', delim=',', header=true, "
        "all_varchar=true, sample_size=-1, ignore_errors=true, null_padding=true)"
    )


def col_names(con) -> list[str]:
    """src_view の列名を元の並び順で返す。"""
    return [d[0] for d in con.execute("SELECT * FROM src_view LIMIT 0").description]


def write_sorted(con, out_path: str) -> None:
    """全行を cycle_id 昇順（同 id 内 timestamp 昇順）で書き出す。"""
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM src_view
            ORDER BY TRY_CAST(cycle_id AS BIGINT) NULLS LAST,
                     TRY_CAST(timestamp AS TIMESTAMP) NULLS LAST
        ) TO '{to_uri(out_path)}' (HEADER, DELIMITER ',')
        """
    )


def write_merged(con, out_path: str, cols: list[str]) -> None:
    """cycle_id ごとに確定行と排出結果行を 1 行へ統合して書き出す。

    各列は「その cycle_id 内で空でない最初の値」を採用（確定行と排出行で
    埋まる列が排他的なため、衝突はほぼ起きない）。"""
    # cycle_id でグルーピング。各列は空文字を無視した代表値（max は空文字より実値を優先）。
    agg = []
    for c in cols:
        if c == "cycle_id":
            continue
        # 空文字を NULL 化してから max（=非空の値）を拾う
        agg.append(f'max(nullif("{c}", \'\')) AS "{c}"')
    select_list = "cycle_id, " + ", ".join(agg)
    # 元の列順を保つ
    ordered = ", ".join(f'"{c}"' for c in cols)
    con.execute(
        f"""
        COPY (
            SELECT {ordered} FROM (
                SELECT cycle_id, {", ".join(agg)}
                FROM src_view
                GROUP BY cycle_id
            )
            ORDER BY TRY_CAST(cycle_id AS BIGINT) NULLS LAST
        ) TO '{to_uri(out_path)}' (HEADER, DELIMITER ',')
        """
    )


def process(con, src: str) -> None:
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    read_view(con, src)
    cols = col_names(con)
    if "cycle_id" not in cols:
        print(f"  [スキップ] {base}: cycle_id 列がありません")
        return

    sorted_out = os.path.join(OUT_DIR, base)
    merged_out = os.path.join(OUT_DIR, f"{stem}_merged{ext}")
    write_sorted(con, sorted_out)
    write_merged(con, merged_out, cols)
    print(f"  {base}")
    print(f"    -> {os.path.relpath(sorted_out, ROOT)}  (昇順・全行)")
    print(f"    -> {os.path.relpath(merged_out, ROOT)}  (昇順・1サイクル1行)")


def main() -> None:
    args = sys.argv[1:]
    if args:
        files = args
    else:
        files = sorted(glob.glob(os.path.join(ROOT, "logs_cycle_5goki", "*.csv")))

    if not files:
        print("対象 CSV が見つかりません。logs_cycle_5goki/*.csv を確認してください。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    for src in files:
        try:
            process(con, src)
        except Exception as e:
            print(f"  [エラー] {os.path.basename(src)}: {e}")
    con.close()
    print("完了。")


if __name__ == "__main__":
    main()
