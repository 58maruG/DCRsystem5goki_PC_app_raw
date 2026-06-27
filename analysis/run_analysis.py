# -*- coding: utf-8 -*-
"""
DCRsystem ログ解析ランナー（DuckDB）

各ログフォルダ（logs_<kind>_5goki）の中の "セッション" ごとに各クエリを実行し、
1 つの Markdown レポート（analysis/report.md）にまとめて出力する。
クエリは A〜E の5カテゴリ（A.レイテンシ・速度 / B.温度・HW相関 / C.メモリリーク /
D.分類品質 / E.障害・健全性）に分類し、その順序・見出しで出力する。

セッション = ファイル名から種別プレフィックス（cycle_ / health_ / detections_ / events_）を
除いた共通サフィックス。例: cycle_20260625.csv / health_20260625.csv → セッション "20260625"。
プレフィックスを持たないファイル（例: 1.csv）はファイル名そのものがセッション名になる。

queries.sql は想定スキーマで書かれた叩き台のため、本スクリプトでは実ファイルの
角括弧付きカラム名（"infer_latency[ms]" など）に合わせてクエリを書き直してある。
実在しない列（serial_rtt_ms / cam_top_temp_c）は除外済み。

使い方:  python analysis/run_analysis.py
"""

from __future__ import annotations

import os
import sys
import glob
import math
import traceback
from datetime import datetime

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

# スクリプトの親（リポジトリルート）を基準にする
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 種別 → (フォルダ, プレフィックス, 拡張子)。dict のキーがビュー名になる。
KINDS = {
    "cyc": ("logs_cycle_5goki", "cycle_", ".csv"),
    "hea": ("logs_health_5goki", "health_", ".csv"),
    "det": ("logs_detections_5goki", "detections_", ".csv"),
    "evt": ("logs_events_5goki", "events_", ".jsonl"),
    #"ben": ("benio_analysis", "", ".csv"),
}


def discover_sessions():
    """全フォルダを走査し、{session: {view: filepath}} を返す。"""
    sessions: dict[str, dict[str, str]] = {}
    for view, (folder, prefix, ext) in KINDS.items():
        d = os.path.join(ROOT, folder)
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, f"*{ext}")):
            name = os.path.basename(path)[: -len(ext)]
            session = name[len(prefix):] if name.startswith(prefix) else name
            sessions.setdefault(session, {})[view] = path
    return sessions


def to_uri(path: str) -> str:
    """DuckDB の文字列リテラル用にパスを整形（バックスラッシュ→スラッシュ）。"""
    return path.replace("\\", "/")


# (key, category, タイトル, 必要ビューの集合, SQL) のリスト。
#   key: 並べ替えに影響されないサマリ抽出用の安定キー。
#   category: 出力レポートの見出し（A〜E）。同じ category を連続させて並べる。
# SQL は cyc/hea/det/evt/ben ビューを参照する。
CAT_A = "A. レイテンシ・速度"
CAT_B = "B. 温度・HW相関"
CAT_C = "C. メモリリーク"
CAT_D = "D. 分類品質"
CAT_E = "E. 障害・健全性"

QUERIES = [
    # ===== A. レイテンシ・速度 =====
    ("q1", CAT_A, "1) 推論レイテンシのスパイク Top20（降クロックとの相関）", {"cyc"}, """
        SELECT cycle_id,
               TRY_CAST("infer_latency[ms]" AS DOUBLE) AS infer_ms,
               TRY_CAST("gpu_clock[mhz]"    AS DOUBLE) AS gpu_clock_mhz,
               TRY_CAST("gpu_temp"       AS DOUBLE) AS gpu_temp_c
        FROM cyc
        WHERE TRY_CAST("infer_latency[ms]" AS DOUBLE) IS NOT NULL
        ORDER BY infer_ms DESC
        LIMIT 20
    """),
    ("q2", CAT_A, "2) 段階別レイテンシの内訳（どの段が律速か）※平均ms", {"cyc"}, """
        SELECT round(avg(TRY_CAST("capture_latency[ms]" AS DOUBLE)), 3) AS capture_ms,
               round(avg(TRY_CAST("preproc[ms]"         AS DOUBLE)), 3) AS preproc_ms,
               round(avg(TRY_CAST("infer_latency[ms]"   AS DOUBLE)), 3) AS infer_ms,
               round(avg(TRY_CAST("postproc[ms]"        AS DOUBLE)), 3) AS postproc_ms
        FROM cyc
    """),
    ("q8", CAT_A, "8) レイテンシのテール分位 + デューティ比（hsv_flag 平均）", {"cyc"}, """
        SELECT round(quantile_cont(infer_ms, 0.50), 3) AS p50,
               round(quantile_cont(infer_ms, 0.95), 3) AS p95,
               round(quantile_cont(infer_ms, 0.99), 3) AS p99,
               round(max(infer_ms), 3)                 AS pmax,
               round(avg(duty), 4)                     AS duty_cycle
        FROM (
            SELECT TRY_CAST("infer_latency[ms]" AS DOUBLE) AS infer_ms,
                   TRY_CAST("hsv_flag"          AS DOUBLE) AS duty
            FROM cyc
        )
    """),
    ("q9", CAT_A, "9) スループット（果実/秒）", {"cyc"}, """
        SELECT round(count(*) / nullif(epoch(max(ts)) - epoch(min(ts)), 0), 4) AS fruits_per_s,
               count(*) AS n
        FROM (SELECT TRY_CAST(timestamp AS TIMESTAMP) AS ts FROM cyc)
    """),
    ("q13", CAT_A, "13) クロック帯ごとのレイテンシ分布（per-cycle、JOIN不要）", {"cyc"}, """
        SELECT CAST((TRY_CAST("gpu_clock[mhz]" AS DOUBLE) // 100) * 100 AS BIGINT) AS clock_bin_mhz,
               count(*) AS n,
               round(avg(TRY_CAST("infer_latency[ms]" AS DOUBLE)), 3) AS inf_avg,
               round(quantile_cont(TRY_CAST("infer_latency[ms]" AS DOUBLE), 0.95), 3) AS inf_p95,
               round(avg(TRY_CAST("gpu_util[%]" AS DOUBLE)), 1) AS gpu_util_avg,
               round(avg(TRY_CAST("gpu_mem_used[mb]" AS DOUBLE)), 1) AS vram_avg
        FROM cyc
        WHERE TRY_CAST("gpu_clock[mhz]" AS DOUBLE) IS NOT NULL
          AND TRY_CAST("infer_latency[ms]" AS DOUBLE) IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """),
    # ===== B. 温度・HW相関 =====
    ("q4", CAT_B, "4) cycle × health を ASOF 結合し相関係数（温度 vs 推論レイテンシ）", {"cyc", "hea"}, """
        SELECT count(*) AS n,
               round(corr(infer_ms, gpu_temp_c), 4) AS corr_infer_gpu_temp,
               round(corr(infer_ms, cpu_temp_c), 4) AS corr_infer_cpu_temp
        FROM (
            SELECT TRY_CAST(c."infer_latency[ms]" AS DOUBLE) AS infer_ms,
                   TRY_CAST(h."gpu_temp"        AS DOUBLE) AS gpu_temp_c,
                   TRY_CAST(h."cpu_temp"        AS DOUBLE) AS cpu_temp_c
            FROM cyc c
            ASOF JOIN hea h
              ON TRY_CAST(c.timestamp AS TIMESTAMP) >= TRY_CAST(h.timestamp AS TIMESTAMP)
        )
    """),
    ("q10", CAT_B, "10) リソース余力（VRAMピーク・使用率p95・電力・最低クロック・最高温）", {"hea"}, """
        SELECT round(max(TRY_CAST("gpu_mem_used[mb]" AS DOUBLE)), 1)              AS vram_peak_mb,
               round(quantile_cont(TRY_CAST("gpu_util[%]" AS DOUBLE), 0.95), 1)  AS gpu_util_p95,
               round(quantile_cont(TRY_CAST("cpu_util[%]" AS DOUBLE), 0.95), 1)  AS cpu_util_p95,
               round(max(TRY_CAST("gpu_power[w]" AS DOUBLE)), 1)                 AS power_peak_w,
               round(min(TRY_CAST("gpu_clock[mhz]" AS DOUBLE)), 0)              AS clock_min_mhz,
               round(max(TRY_CAST("gpu_temp" AS DOUBLE)), 1)                 AS temp_max_c
        FROM hea
    """),
    # ===== C. メモリリーク =====
    ("q11", CAT_C, "11) リーク検出：時間あたり増加（warmup 先頭120秒を除外）", {"hea"}, """
        WITH base AS (
            SELECT TRY_CAST(timestamp AS TIMESTAMP)            AS ts,
                   TRY_CAST("proc_rss[mb]" AS DOUBLE)          AS rss,
                   TRY_CAST("torch_vram_alloc[mb]" AS DOUBLE)  AS valloc,
                   TRY_CAST("torch_vram_reserved[mb]" AS DOUBLE) AS vreserved,
                   TRY_CAST("disk_free[gb]" AS DOUBLE)         AS disk
            FROM hea
        )
        SELECT round(regr_slope(rss,       epoch(ts)) * 3600, 3) AS ram_mb_per_h,
               round(regr_slope(valloc,    epoch(ts)) * 3600, 3) AS vram_alloc_mb_per_h,
               round(regr_slope(vreserved, epoch(ts)) * 3600, 3) AS vram_reserved_mb_per_h,
               round(regr_slope(disk,      epoch(ts)) * 3600, 4) AS disk_gb_per_h
        FROM base
        WHERE ts > (SELECT min(ts) + INTERVAL 120 SECOND FROM base)
    """),
    ("q12", CAT_C, "12) サイクルあたりの漏れ（cycles_total で正規化）", {"hea"}, """
        SELECT round(regr_slope(TRY_CAST("proc_rss[mb]" AS DOUBLE),
                                TRY_CAST("cycles_total[n]" AS DOUBLE)), 4) AS mb_per_cycle
        FROM hea
        WHERE TRY_CAST("cycles_total[n]" AS DOUBLE) > 0
    """),
    # ===== D. 分類品質 =====
    ("q5", CAT_D, "5) 検出クラス別の集計（弱クラスの確認）", {"det"}, """
        SELECT class,
               count(*) AS n,
               round(avg(TRY_CAST(conf_ave AS DOUBLE)), 3) AS conf_ave,
               sum(TRY_CAST(final_flag AS INTEGER)) AS finals
        FROM det
        GROUP BY class
        ORDER BY n DESC
    """),
    ("qB1", CAT_D, "B1) benio_analysis クラス別集計（弱クラスの確認）", {"ben"}, """
        SELECT class,
               count(*) AS n,
               round(avg(TRY_CAST(conf_ave AS DOUBLE)), 3) AS conf_ave,
               sum(TRY_CAST(final_flag AS INTEGER)) AS finals
        FROM ben
        GROUP BY class
        ORDER BY n DESC
    """),
    # ===== E. 障害・健全性 =====
    ("q3", CAT_E, "3) outcome_flag で失敗を切り分け（検証データのみ値が入る）", {"cyc"}, """
        SELECT CAST("outcome_flag" AS VARCHAR) AS outcome,
               count(*) AS n,
               round(avg(TRY_CAST("infer_latency[ms]" AS DOUBLE)), 3) AS infer_ms,
               round(avg(TRY_CAST("eject_delay[ms]"   AS DOUBLE)), 3) AS eject_delay_ms
        FROM cyc
        WHERE "outcome_flag" IS NOT NULL AND CAST("outcome_flag" AS VARCHAR) <> ''
        GROUP BY 1
        ORDER BY n DESC
    """),
    ("q6", CAT_E, "6) イベント（JSONL）の型 × 重大度ごとの発生数", {"evt"}, """
        SELECT type, sev, count(*) AS n
        FROM evt
        GROUP BY type, sev
        ORDER BY n DESC
    """),
    ("q7", CAT_E, "7) ロガーのドロップ検知（dropped_logs / queue_depth の最大値）", {"hea"}, """
        SELECT max(TRY_CAST("dropped_logs[n]" AS BIGINT)) AS dropped_max,
               max(TRY_CAST("queue_depth[n]"  AS BIGINT)) AS queue_depth_max,
               count(*) AS samples
        FROM hea
    """),
]


def make_views(con, files: dict[str, str]):
    """セッションの各ファイルをビュー化。作成できたビュー名の集合を返す。"""
    available = set()
    for view, path in files.items():
        uri = to_uri(path)
        try:
            if view == "evt":
                con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_json_auto('{uri}', format='newline_delimited', "
                    f"union_by_name=true, ignore_errors=true)"
                )
            else:
                # 区切り文字とヘッダを明示し、全列 VARCHAR で読む。
                # 自動検出に任せると一部ファイルで区切り推定が外れて
                # ヘッダ全体が「1列」になることがあるため。数値化は各クエリの
                # TRY_CAST で行う。short row は null_padding、壊れた行は ignore_errors で吸収。
                con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_csv('{uri}', delim=',', header=true, "
                    f"all_varchar=true, sample_size=-1, ignore_errors=true, "
                    f"null_padding=true)"
                )
            # 中身が読めるか軽く確認
            con.execute(f"SELECT 1 FROM {view} LIMIT 1")
            available.add(view)
        except Exception as e:
            print(f"  [警告] {view} のビュー作成に失敗: {os.path.basename(path)} -> {e}")
    return available


def drop_views(con):
    for view in KINDS:
        try:
            con.execute(f"DROP VIEW IF EXISTS {view}")
        except Exception:
            pass


def fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if v == int(v):
            return str(int(v))
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def md_table(cols, rows) -> str:
    if not rows:
        return "_（該当データなし）_\n"
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(fmt_cell(c) for c in r) + " |" for r in rows)
    return f"{header}\n{sep}\n{body}\n"


def run_query(con, sql):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def main():
    sessions = discover_sessions()
    if not sessions:
        print("ログファイルが見つかりませんでした。logs_*_5goki フォルダを確認してください。")
        return

    con = duckdb.connect(database=":memory:")
    out_lines = []
    out_lines.append("# DCRsystem ログ解析レポート\n")
    out_lines.append(f"生成日時: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    out_lines.append(f"対象セッション数: {len(sessions)}\n")

    # サマリ用に主要指標を集める
    summary = []

    for session in sorted(sessions.keys()):
        files = sessions[session]
        print(f"[解析] セッション '{session}'  ファイル: {', '.join(sorted(files))}")
        out_lines.append(f"\n---\n\n## セッション: `{session}`\n")
        present = ", ".join(f"{v}={os.path.basename(p)}" for v, p in sorted(files.items()))
        out_lines.append(f"対象ファイル: {present}\n")

        available = make_views(con, files)

        results = {}  # key -> (cols, rows)
        last_cat = None
        for key, category, title, needs, sql in QUERIES:
            # 必要ファイルが揃わないクエリは丸ごとスキップ（見出しも出さない）。
            #   これで ben のみのセッションなどで「必要なファイルがありません」の
            #   空セクションが大量に並ぶのを防ぐ。
            if needs - available:
                continue
            # カテゴリが変わったら見出し（A〜E）を1度だけ出す。
            #   スキップで空になったカテゴリの見出しはここに到達しないので出ない。
            if category != last_cat:
                out_lines.append(f"\n### {category}\n")
                last_cat = category
            out_lines.append(f"\n#### {title}\n")
            try:
                cols, rows = run_query(con, sql)
                results[key] = (cols, rows)
                out_lines.append(md_table(cols, rows))
            except Exception as e:
                out_lines.append(f"_（クエリ失敗: {e}）_\n")
                print(f"  [エラー] {title}: {e}")

        # サマリ指標の抽出（安定キーで参照。並べ替えの影響を受けない）
        def pick(key, col):
            if key not in results:
                return None
            cols, rows = results[key]
            if not rows or col not in cols:
                return None
            return rows[0][cols.index(col)]

        metrics = (
            pick("q8", "p50"),             # テール分位
            pick("q8", "p95"),
            pick("q9", "fruits_per_s"),    # スループット
            pick("q10", "vram_peak_mb"),   # リソース余力
            pick("q10", "temp_max_c"),
            pick("q11", "ram_mb_per_h"),   # リーク
            pick("q7", "dropped_max"),     # ロガードロップ
        )
        # 主要指標が1つも無いセッション（ben のみ等）はサマリ表に載せない
        if any(m is not None for m in metrics):
            summary.append((session, *metrics))

        drop_views(con)

    # 冒頭サマリ表を組み立ててヘッダ直後に挿入
    sum_cols = ["session", "infer_p50_ms", "infer_p95_ms", "fruits/s",
                "vram_peak_mb", "temp_max_c", "ram_mb/h", "dropped_max"]
    summary_md = "\n## 横断サマリ（主要指標）\n\n" + md_table(sum_cols, summary)
    out_lines.insert(3, summary_md)

    report_path = os.path.join(ROOT, "analysis", "report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
    except OSError as e:
        print(f"レポート書き込みに失敗しました: {e}")
        con.close()
        return

    con.close()

    print("\n===== 横断サマリ =====")
    print(md_table(sum_cols, summary))
    print(f"レポートを書き出しました: {os.path.relpath(report_path, ROOT)}")


if __name__ == "__main__":
    main()
