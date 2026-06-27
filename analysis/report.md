# DCRsystem ログ解析レポート

生成日時: 2026-06-27 00:38:09

対象セッション数: 1


## 横断サマリ（主要指標）

| session | infer_p50_ms | infer_p95_ms | fruits/s | vram_peak_mb | temp_max_c | ram_mb/h | dropped_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260623 | 14.49 | 15.96 | 0.7806 | 2429.7 | 58 | -252.623 | 0 |


---

## セッション: `20260623`

対象ファイル: cyc=cycle_20260623.csv, det=detections_20260623.csv, evt=events_20260623.jsonl, hea=health_20260623.csv


### A. レイテンシ・速度


#### 1) 推論レイテンシのスパイク Top20（降クロックとの相関）

| cycle_id | infer_ms | gpu_clock_mhz | gpu_temp_c |
| --- | --- | --- | --- |
| 130 | 20.34 | 2040 | 58 |
| 304 | 20.16 | 2040 | 57 |
| 609 | 20.07 | 2040 | 57 |
| 515 | 19.3 | 2040 | 57 |
| 492 | 18.57 | 2040 | 57 |
| 586 | 18.56 | 2040 | 57 |
| 128 | 16.81 | 2040 | 58 |
| 93 | 16.77 | 2040 | 56 |
| 448 | 16.62 | 2040 | 57 |
| 370 | 16.55 | 2040 | 57 |
| 385 | 16.48 | 2040 | 57 |
| 652 | 16.42 | 2040 | 57 |
| 548 | 16.38 | 2040 | 57 |
| 474 | 16.33 | 2040 | 57 |
| 371 | 16.32 | 2040 | 57 |
| 117 | 16.3 | 2040 | 57 |
| 311 | 16.25 | 2040 | 57 |
| 233 | 16.25 | 2040 | 57 |
| 278 | 16.22 | 2040 | 57 |
| 514 | 16.22 | 2040 | 57 |


#### 2) 段階別レイテンシの内訳（どの段が律速か）※平均ms

| capture_ms | preproc_ms | infer_ms | postproc_ms |
| --- | --- | --- | --- |
| 48.759 | 0.674 | 14.518 | 0.329 |


#### 8) レイテンシのテール分位 + デューティ比（hsv_flag 平均）

| p50 | p95 | p99 | pmax | duty_cycle |
| --- | --- | --- | --- | --- |
| 14.49 | 15.96 | 16.788 | 20.34 | 1 |


#### 9) スループット（果実/秒）

| fruits_per_s | n |
| --- | --- |
| 0.7806 | 1312 |


#### 13) クロック帯ごとのレイテンシ分布（per-cycle、JOIN不要）

| clock_bin_mhz | n | inf_avg | inf_p95 | gpu_util_avg | vram_avg |
| --- | --- | --- | --- | --- | --- |
| 2040 | 656 | 14.518 | 15.96 | 7.9 | 2338.9 |


### B. 温度・HW相関


#### 4) cycle × health を ASOF 結合し相関係数（温度 vs 推論レイテンシ）

| n | corr_infer_gpu_temp | corr_infer_cpu_temp |
| --- | --- | --- |
| 1312 | 0.2631 | 0.0075 |


#### 10) リソース余力（VRAMピーク・使用率p95・電力・最低クロック・最高温）

| vram_peak_mb | gpu_util_p95 | cpu_util_p95 | power_peak_w | clock_min_mhz | temp_max_c |
| --- | --- | --- | --- | --- | --- |
| 2429.7 | 12.2 | 6.2 | 44.3 | 2040 | 58 |


### C. メモリリーク


#### 11) リーク検出：時間あたり増加（warmup 先頭120秒を除外）

| ram_mb_per_h | vram_alloc_mb_per_h | vram_reserved_mb_per_h | disk_gb_per_h |
| --- | --- | --- | --- |
| -252.623 | 0 | 0 | -0.2344 |


#### 12) サイクルあたりの漏れ（cycles_total で正規化）

| mb_per_cycle |
| --- |
| -0.1627 |


### D. 分類品質


#### 5) 検出クラス別の集計（弱クラスの確認）

| class | n | conf_ave | finals |
| --- | --- | --- | --- |
| healthy | 493 | 0.868 | 373 |
| unripe | 394 | 0.891 | 141 |
| malformation | 186 | 0.866 | 137 |
| twin | 40 | 0.772 | 0 |
| blacktwin | 29 | 0.811 | 5 |


### E. 障害・健全性


#### 3) outcome_flag で失敗を切り分け（検証データのみ値が入る）

| outcome | n | infer_ms | eject_delay_ms |
| --- | --- | --- | --- |
| 1 | 656 |  | 6.24 |


#### 6) イベント（JSONL）の型 × 重大度ごとの発生数

| type | sev | n |
| --- | --- | --- |
| arduino_heartbeat | INFO | 85 |
| mem_snapshot | INFO | 42 |
| gui_state | INFO | 2 |
| arduino_params | INFO | 1 |
| shutdown | INFO | 1 |
| estop_cleared | INFO | 1 |
| camera_error | ERROR | 1 |
| arduino_connect | INFO | 1 |
| startup | INFO | 1 |


#### 7) ロガーのドロップ検知（dropped_logs / queue_depth の最大値）

| dropped_max | queue_depth_max | samples |
| --- | --- | --- |
| 0 | 80 | 2557 |
