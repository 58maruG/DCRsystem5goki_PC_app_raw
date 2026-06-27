-- DCRsystem ログ解析クエリ集（DuckDB）
-- インストール:  pip install duckdb  /  duckdb CLI でも可
-- 実行例:        duckdb -c ".read analysis/queries.sql"
-- 実フォルダは logs_<kind>_5goki/<kind>_YYYYMMDD.{csv,jsonl}。line 名(5goki)は環境に合わせて変更。
-- ワイルドカードで日付分割ファイルをまとめて読む。

-- 1) 推論レイテンシのスパイク Top20（降クロックとの相関を見る）
SELECT cycle_id, infer_latency_ms, gpu_clock_mhz, gpu_temp_c
FROM 'logs_cycle_5goki/cycle_*.csv'
ORDER BY infer_latency_ms DESC
LIMIT 20;

-- 2) 段階別レイテンシの内訳（どの段が律速か）
SELECT
  avg(capture_latency_ms) cap, avg(preproc_ms) pre,
  avg(infer_latency_ms) inf, avg(postproc_ms) post, avg(serial_rtt_ms) rtt
FROM 'logs_cycle_5goki/cycle_*.csv';

-- 3) 失敗を三層で切り分け（outcome が入っている検証データのみ）
SELECT outcome, count(*) n,
       avg(infer_latency_ms) inf, avg(eject_delay_ms) dly
FROM 'logs_cycle_5goki/cycle_*.csv'
WHERE outcome <> ''
GROUP BY outcome;

-- 4) cycle × health を時刻で ASOF 結合（温度 vs レイテンシ）
SELECT c.cycle_id, c.infer_latency_ms, h.gpu_temp_c, h.cam_top_temp_c, h.cpu_temp_c
FROM 'logs_cycle_5goki/cycle_*.csv' c
ASOF JOIN 'logs_health_5goki/health_*.csv' h
  ON c.timestamp >= h.timestamp;

-- 5) 多ラベル検出のクラス別集計（弱クラスの確認）
SELECT class, count(*) n, avg(confidence) conf, sum(is_final) finals
FROM 'benio_analysis/*.csv'
GROUP BY class
ORDER BY n DESC;

-- 6) イベント（JSONL）をそのまま読む。型ごとの発生数
SELECT type, sev, count(*) n
FROM read_json_auto('logs_events_5goki/events_*.jsonl')
GROUP BY type, sev
ORDER BY n DESC;

-- 7) ロガーのドロップ検知（health の dropped_logs が増えていないか）
SELECT timestamp, dropped_logs, queue_depth
FROM 'logs_health_5goki/health_*.csv'
ORDER BY timestamp DESC
LIMIT 20;

-- ============================================================
-- v1.1 追加クエリ（HWダウングレード判断 / メモリ・リソースリーク検出）
-- ============================================================

-- ===== HWダウングレード判断 =====
-- 8) レイテンシのテール + デューティ比（果実が来た割合）
SELECT quantile_cont(infer_latency_ms, 0.50) p50,
       quantile_cont(infer_latency_ms, 0.95) p95,
       quantile_cont(infer_latency_ms, 0.99) p99,
       max(infer_latency_ms) pmax,
       avg(hsv_pass) duty_cycle
FROM 'logs_cycle_5goki/cycle_*.csv';

-- 9) スループット（果実/秒）= レイテンシ予算の分母
SELECT count(*) / (epoch(max(timestamp::timestamp)) - epoch(min(timestamp::timestamp))) fruits_per_s
FROM 'logs_cycle_5goki/cycle_*.csv';

-- 10) 余力の核心：VRAMピーク・使用率p95・電力・最小クロック・最高温
SELECT max(gpu_mem_used_mb) vram_peak_mb,
       quantile_cont(gpu_util, 0.95) gpu_util_p95,
       quantile_cont(cpu_util, 0.95) cpu_util_p95,
       max(gpu_power_w) power_peak_w,
       min(gpu_clock_mhz) clock_min,
       max(gpu_temp_c) temp_max
FROM 'logs_health_5goki/health_*.csv';

-- ===== リーク検出 =====
-- 11) 時間あたりの増加（正で持続ならリーク）。warmup（先頭120秒）は除外
SELECT regr_slope(proc_rss_mb,            epoch(timestamp::timestamp)) * 3600 ram_mb_per_h,
       regr_slope(torch_vram_alloc_mb,    epoch(timestamp::timestamp)) * 3600 vram_alloc_mb_per_h,
       regr_slope(torch_vram_reserved_mb, epoch(timestamp::timestamp)) * 3600 vram_reserved_mb_per_h,
       regr_slope(disk_free_gb,           epoch(timestamp::timestamp)) * 3600 disk_gb_per_h
FROM 'logs_health_5goki/health_*.csv'
WHERE timestamp::timestamp > (SELECT min(timestamp::timestamp) + INTERVAL 120 SECOND
                              FROM 'logs_health_5goki/health_*.csv');

-- 12) サイクルあたりの漏れ（cycles_total で正規化）
SELECT regr_slope(proc_rss_mb, cycles_total) mb_per_cycle
FROM 'logs_health_5goki/health_*.csv'
WHERE cycles_total > 0;

-- 13) per-cycle 相関（JOIN不要）：果実ごとのレイテンシ ⇄ その瞬間のHW状態。
--     cycle 自身に HW負荷系スナップショット（health 1Hz の写し）を持たせたので
--     health と ASOF JOIN せずに直接相関が見られる。クロック帯ごとのレイテンシ分布。
SELECT (gpu_clock_mhz // 100) * 100 AS clock_bin_mhz,
       count(*) n,
       avg(infer_latency_ms) inf_avg,
       quantile_cont(infer_latency_ms, 0.95) inf_p95,
       avg(gpu_util) gpu_util_avg,
       avg(gpu_mem_used_mb) vram_avg
FROM 'logs_cycle_5goki/cycle_*.csv'
WHERE gpu_clock_mhz <> '' AND infer_latency_ms <> ''
GROUP BY clock_bin_mhz
ORDER BY clock_bin_mhz;

-- ============================================================
-- cycle_id 昇順で見る（ファイル上は確定行と排出結果行が時間差で混ざり前後する）
-- ============================================================

-- 14) 全行を cycle_id 昇順（同 id 内は timestamp 昇順）で表示
SELECT *
FROM 'logs_cycle_5goki/cycle_*.csv'
ORDER BY TRY_CAST(cycle_id AS BIGINT), TRY_CAST(timestamp AS TIMESTAMP);

-- 15) 1 サイクル 1 行に統合（確定行＋排出結果行を cycle_id でまとめる）
--     各列は空文字を無視した代表値を採用（確定行と排出行で埋まる列は排他的）。
SELECT cycle_id,
       max(nullif(timestamp, ''))            AS timestamp,
       max(nullif("infer_latency[ms]", ''))  AS "infer_latency[ms]",
       max(nullif("eject_flag", ''))         AS eject_flag,
       max(nullif("planned_eject_ts", ''))   AS planned_eject_ts,
       max(nullif("eject_delay[ms]", ''))    AS "eject_delay[ms]",
       max(nullif("outcome_flag", ''))       AS outcome_flag
FROM 'logs_cycle_5goki/cycle_*.csv'
GROUP BY cycle_id
ORDER BY TRY_CAST(cycle_id AS BIGINT);

-- 解釈の注意:
--   torch_vram_alloc が単調増加 = 本物のテンソル参照リーク。
--   reserved だけ増えて alloc 横ばいはアロケータ確保で無害。
--   決定打は「アイドル区間（cycle 行が無い時間帯）でも下がらない」こと。
--   cycle の HW列は「果実が来た瞬間（負荷時）」の写し、health は「アイドル含む連続」。
--   同じ指標でも時間軸が違うので、ダウングレード判断では両方を併用する。
