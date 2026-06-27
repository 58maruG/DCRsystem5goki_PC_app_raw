# -*- coding: utf-8 -*-
"""
DCRsystem ログ可視化（matplotlib / 静的PNG）

run_analysis.py が数表(Markdown)で出す指標を、グラフにして
analysis/figures/<セッション>/ に PNG で書き出す補助スクリプト。
DuckDB でCSVを読み込み(角括弧付きカラム名・全列VARCHAR→TRY_CAST)、
matplotlib で描画する。queries.sql / run_analysis.py には手を入れない。

出力する図（セッションごと・run_analysis.py の A〜E カテゴリに対応）:
  A_latency_speed.png … 推論レイテンシの時系列・分布・段階別平均・スループット
  B_temp_clock.png    … GPU温度/クロック vs 推論レイテンシ の散布
  C_memory_leak.png   … proc_rss / torch_vram の時系列＋回帰直線（リーク）
  D_class.png         … クラス別の検出件数／確定件数(final_flag)＋平均conf（弱クラス確認）

使い方:  python analysis/plot_analysis.py
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

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # 画面を持たない環境でも保存できるよう非対話バックエンド
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("matplotlib / numpy が見つかりません。`pip install matplotlib` を実行してください。")
    sys.exit(1)

# Windows の日本語フォント（無ければ DejaVu にフォールバック＝英数字のみ）
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = True

# Windows コンソールの文字化け対策
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 種別 → (フォルダ, プレフィックス, 拡張子)。run_analysis.py と同じ規則。
KINDS = {
    "cyc": ("logs_cycle_5goki", "cycle_", ".csv"),
    "hea": ("logs_health_5goki", "health_", ".csv"),
    "det": ("logs_detections_5goki", "detections_", ".csv"),
}


def discover_sessions() -> dict[str, dict[str, str]]:
    """{session: {view: filepath}} を返す。"""
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


def make_views(con, files: dict[str, str]) -> set[str]:
    """各CSVをDuckDBビュー化。作成できたビュー名の集合を返す。"""
    available = set()
    for view, path in files.items():
        uri = path.replace("\\", "/")
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_csv('{uri}', delim=',', header=true, "
                f"all_varchar=true, sample_size=-1, ignore_errors=true, null_padding=true)"
            )
            con.execute(f"SELECT 1 FROM {view} LIMIT 1")
            available.add(view)
        except Exception as e:
            print(f"  [警告] {view} 読み込み失敗: {os.path.basename(path)} -> {e}")
    return available


def fetch_cols(con, sql: str) -> dict[str, list]:
    """SQL結果を {列名: [値,...]} で返す。"""
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    data: dict[str, list] = {c: [] for c in cols}
    for row in cur.fetchall():
        for c, v in zip(cols, row):
            data[c].append(v)
    return data


def _xy(xs, ys):
    """両方 None でないペアだけ取り出す。"""
    px, py = [], []
    for a, b in zip(xs, ys):
        if a is not None and b is not None:
            px.append(a)
            py.append(b)
    return px, py


def _clean(vs):
    return [v for v in vs if v is not None]


# ============================================================
# 図 A: レイテンシ・速度（時系列 / 分布 / 段階別平均 / スループット）
# ============================================================
def plot_a_latency_speed(con, figdir: str):
    d = fetch_cols(con, """
        SELECT TRY_CAST(timestamp AS TIMESTAMP)            AS ts,
               TRY_CAST("infer_latency[ms]"   AS DOUBLE)   AS infer_ms,
               TRY_CAST("capture_latency[ms]" AS DOUBLE)   AS capture_ms,
               TRY_CAST("preproc[ms]"         AS DOUBLE)   AS preproc_ms,
               TRY_CAST("postproc[ms]"        AS DOUBLE)   AS postproc_ms
        FROM cyc ORDER BY ts
    """)
    ts, infer = _xy(d["ts"], d["infer_ms"])
    vals = _clean(d["infer_ms"])
    if not vals:
        print("  [skip] A: infer_latency が空")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) 推論レイテンシ 時系列
    ax = axes[0][0]
    if ts:
        ax.plot(ts, infer, lw=0.8, color="#1f77b4")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.plot(infer, lw=0.8, color="#1f77b4")
    p95 = float(np.percentile(vals, 95))
    ax.axhline(p95, color="#d62728", ls="--", lw=1, label=f"p95={p95:.1f}ms")
    ax.set_title("推論レイテンシ 時系列")
    ax.set_xlabel("時刻")
    ax.set_ylabel("infer_latency [ms]")
    ax.legend(loc="upper right", fontsize=9)

    # (2) 推論レイテンシ 分布＋分位
    ax = axes[0][1]
    ax.hist(vals, bins=40, color="#1f77b4", alpha=0.8)
    for q, c in ((50, "#2ca02c"), (95, "#ff7f0e"), (99, "#d62728")):
        v = float(np.percentile(vals, q))
        ax.axvline(v, color=c, ls="--", lw=1.2, label=f"p{q}={v:.1f}")
    ax.set_title("推論レイテンシ 分布")
    ax.set_xlabel("infer_latency [ms]")
    ax.set_ylabel("件数")
    ax.legend(fontsize=9)

    # (3) 段階別 平均ms（どの段が律速か）
    ax = axes[1][0]
    stages = [("capture", d["capture_ms"]), ("preproc", d["preproc_ms"]),
              ("infer", d["infer_ms"]), ("postproc", d["postproc_ms"])]
    names, means = [], []
    for nm, col in stages:
        c = _clean(col)
        if c:
            names.append(nm)
            means.append(float(np.mean(c)))
    bars = ax.bar(names, means, color=["#8c564b", "#9467bd", "#1f77b4", "#e377c2"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("段階別 平均レイテンシ")
    ax.set_ylabel("平均 [ms]")

    # (4) スループット（件/分）
    ax = axes[1][1]
    t = fetch_cols(con, """
        SELECT date_trunc('minute', TRY_CAST(timestamp AS TIMESTAMP)) AS m,
               count(*) AS n
        FROM cyc
        WHERE TRY_CAST(timestamp AS TIMESTAMP) IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)
    m, n = _xy(t["m"], t["n"])
    n = [int(v) for v in n]
    if m:
        ax.plot(m, n, "o-", color="#2ca02c", lw=1.0, ms=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        avg = float(np.mean(n))
        ax.axhline(avg, color="#888888", ls="--", lw=1, label=f"平均={avg:.1f} 件/分")
        ax.legend(fontsize=9)
    ax.set_title("スループット（件/分）")
    ax.set_xlabel("時刻")
    ax.set_ylabel("確定果実数 / 分")

    out = os.path.join(figdir, "A_latency_speed.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [出力] {os.path.relpath(out, ROOT)}")


# ============================================================
# 図 B: 温度/クロック vs 推論レイテンシ
# ============================================================
def plot_b_temp_clock(con, figdir: str):
    d = fetch_cols(con, """
        SELECT TRY_CAST("gpu_temp"          AS DOUBLE) AS gpu_temp,
               TRY_CAST("gpu_clock[mhz]"    AS DOUBLE) AS gpu_clock,
               TRY_CAST("infer_latency[ms]" AS DOUBLE) AS infer_ms
        FROM cyc
    """)
    if not _clean(d["infer_ms"]):
        print("  [skip] temp_clock: infer_latency が空")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, key, label in ((axes[0], "gpu_temp", "GPU温度 [℃]"),
                           (axes[1], "gpu_clock", "GPUクロック [MHz]")):
        x, y = _xy(d[key], d["infer_ms"])
        if x:
            ax.scatter(x, y, s=8, alpha=0.3, color="#1f77b4", edgecolors="none")
            # 分散ゼロ（値が一定）だと相関は未定義。0除算warningを避けて n/a 表示にする。
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                r_txt = f"r={float(np.corrcoef(x, y)[0, 1]):.3f}"
            else:
                r_txt = "r=n/a"
            ax.set_title(f"{label} vs 推論レイテンシ（相関 {r_txt}）")
        else:
            ax.set_title(f"{label}（データなし）")
        ax.set_xlabel(label)
        ax.set_ylabel("infer_latency [ms]")

    out = os.path.join(figdir, "B_temp_clock.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [出力] {os.path.relpath(out, ROOT)}")


# ============================================================
# 図 C: メモリ / VRAM リーク（時系列＋回帰直線）
# ============================================================
def plot_c_memory_leak(con, figdir: str):
    d = fetch_cols(con, """
        SELECT TRY_CAST(timestamp AS TIMESTAMP)              AS ts,
               TRY_CAST("proc_rss[mb]" AS DOUBLE)            AS rss,
               TRY_CAST("torch_vram_alloc[mb]" AS DOUBLE)    AS valloc,
               TRY_CAST("torch_vram_reserved[mb]" AS DOUBLE) AS vreserved
        FROM hea ORDER BY ts
    """)
    if not _clean(d["rss"]) and not _clean(d["valloc"]):
        print("  [skip] memory_leak: health 数値が空")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    def _fit_line(ax, ts, ys, color, label):
        x, y = _xy(ts, ys)
        if not x:
            return
        ax.plot(x, y, lw=0.9, color=color, label=label)
        # 経過秒で1次回帰し mb/h を出す
        t0 = x[0]
        secs = np.array([(t - t0).total_seconds() for t in x], dtype=float)
        if len(secs) > 1 and secs[-1] > 0:
            a, b = np.polyfit(secs, np.array(y, dtype=float), 1)
            ax.plot(x, a * secs + b, ls="--", lw=1.2, color=color, alpha=0.7)
            ax.text(0.02, 0.95 - 0.07 * ax._fit_i, f"{label}: {a*3600:.2f} MB/h",
                    transform=ax.transAxes, color=color, fontsize=9, va="top")
            ax._fit_i += 1

    # proc_rss
    ax = axes[0]
    ax._fit_i = 0
    _fit_line(ax, d["ts"], d["rss"], "#d62728", "proc_rss")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("プロセスRSS の推移（リーク検出）")
    ax.set_xlabel("時刻")
    ax.set_ylabel("proc_rss [MB]")
    ax.legend(loc="lower right", fontsize=9)

    # torch vram
    ax = axes[1]
    ax._fit_i = 0
    _fit_line(ax, d["ts"], d["valloc"], "#1f77b4", "vram_alloc")
    _fit_line(ax, d["ts"], d["vreserved"], "#ff7f0e", "vram_reserved")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("torch VRAM の推移（リーク検出）")
    ax.set_xlabel("時刻")
    ax.set_ylabel("VRAM [MB]")
    ax.legend(loc="lower right", fontsize=9)

    out = os.path.join(figdir, "C_memory_leak.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [出力] {os.path.relpath(out, ROOT)}")


# ============================================================
# 図 D: 分類品質（クラス別 検出件数＋平均conf）
# ============================================================
def plot_d_class(con, figdir: str):
    d = fetch_cols(con, """
        SELECT class,
               count(*) AS n,
               sum(TRY_CAST(final_flag AS INTEGER)) AS finals,
               round(avg(TRY_CAST(conf_ave AS DOUBLE)), 3) AS conf_ave,
               round(avg(CASE WHEN TRY_CAST(final_flag AS INTEGER) = 1
                              THEN TRY_CAST(conf_ave AS DOUBLE) END), 3) AS conf_final
        FROM det
        WHERE class IS NOT NULL AND class <> '' AND class <> 'None'
        GROUP BY class ORDER BY n DESC
    """)
    classes = d["class"]
    if not classes:
        print("  [skip] D: detections クラスが空")
        return
    ns = [int(v) if v is not None else 0 for v in d["n"]]            # 検出件数（クラス検出のあった個体数）
    finals = [int(v) if v is not None else 0 for v in d["finals"]]  # 確定件数（final_flag=1）
    confs = [float(v) if v is not None else 0.0 for v in d["conf_ave"]]          # 全検出の平均conf
    # 確定（final_flag=1）のみの平均conf。確定0件のクラスは conf=0 として■を打つ。
    confs_final = [float(v) if v is not None else 0.0 for v in d["conf_final"]]

    x = np.arange(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    # 左軸: 検出件数 と 確定件数 を並び棒で
    b1 = ax.bar(x - width / 2, ns, width, color="#1f77b4", alpha=0.85, label="検出件数")
    b2 = ax.bar(x + width / 2, finals, width, color="#2ca02c", alpha=0.85, label="確定件数")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h, str(int(h)),
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("件数")
    ax.set_title("クラス別 検出件数 / 確定件数 と 平均conf")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")

    # 右軸: 平均conf。全検出（赤）と 確定のみ（橙）の2本を重ねる。
    ax2 = ax.twinx()
    ln = ax2.plot(x, confs, "o-", color="#d62728", lw=1.2, label="平均conf(全検出)")
    # 確定のみ: 確定0件のクラスも conf=0 として■を打ち、全クラスを破線でつなぐ。
    ln_f = ax2.plot(x, confs_final, "s--", color="#ff7f0e", lw=1.2, label="平均conf(確定のみ)")
    ax2.set_ylabel("平均conf", color="#d62728")
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # 凡例はグラフに被らないよう右外側へ出す（保存時 bbox_inches='tight' で見切れ防止）
    handles = [b1, b2, ln[0], ln_f[0]]
    ax.legend(handles, [h.get_label() for h in handles],
              loc="center left", bbox_to_anchor=(1.12, 0.5), fontsize=9)

    out = os.path.join(figdir, "D_class.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [出力] {os.path.relpath(out, ROOT)}")


def main():
    sessions = discover_sessions()
    if not sessions:
        print("ログファイルが見つかりませんでした。logs_*_5goki フォルダを確認してください。")
        return

    con = duckdb.connect(database=":memory:")
    for session in sorted(sessions.keys()):
        files = sessions[session]
        print(f"[描画] セッション '{session}'  ファイル: {', '.join(sorted(files))}")
        available = make_views(con, files)

        figdir = os.path.join(ROOT, "analysis", "figures", session)
        os.makedirs(figdir, exist_ok=True)

        try:
            if "cyc" in available:
                plot_a_latency_speed(con, figdir)   # A. レイテンシ・速度
                plot_b_temp_clock(con, figdir)      # B. 温度・HW相関
            if "hea" in available:
                plot_c_memory_leak(con, figdir)     # C. メモリリーク
            if "det" in available:
                plot_d_class(con, figdir)           # D. 分類品質
        except Exception as e:
            import traceback
            print(f"  [エラー] セッション '{session}' の描画に失敗: {e}")
            traceback.print_exc()

        for view in KINDS:
            con.execute(f"DROP VIEW IF EXISTS {view}")

    con.close()
    print(f"\n完了。PNGは {os.path.join('analysis', 'figures')}/<セッション>/ に出力しました。")


if __name__ == "__main__":
    main()
