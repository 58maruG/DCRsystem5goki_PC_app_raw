"""6コアに制限した環境で、スレッド構成ごとの実挙動を比較する。
Jetson Orin Nano (6コア) のコア数制約を CPU affinity で再現する。
※ARMコアの絶対性能は再現できないので、見るべきは「構成間の相対差」と
  「ネストした並列によるオーバーサブスクリプションの有無」。
"""
import json, os, pathlib, sys, time, threading, statistics
import numpy as np, cv2, psutil

# リポジトリルート（このファイルの1つ上）を基準にする
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

PROC = psutil.Process(os.getpid())
# Jetson Orin Nano の 6 コアを再現する。実機が6コア以下ならそのまま全コアを使う。
try:
    if os.cpu_count() and os.cpu_count() > 6:
        PROC.cpu_affinity([0, 1, 2, 3, 4, 5])
except Exception as e:
    print(f"[warn] cpu_affinity を設定できません（全コアで計測します）: {e}")

CAMS = [("cam_top", 640), ("cam_under", 640), ("cam_inside", 560), ("cam_outside", 500)]
PERIOD = 0.050        # 20fps = QTimer 50ms
DURATION = 6.0        # 各構成の計測時間(秒)
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def make_frame(size):
    f = np.random.randint(0, 60, (size, size, 3), dtype=np.uint8)
    cv2.circle(f, (size // 2, size // 2), size // 6, (40, 40, 220), -1)
    return f

JOBS = []
for cam, size in CAMS:
    with open(f"json/hsv_config_{cam}.json") as f:
        cfg = json.load(f)
    JOBS.append((cam, make_frame(size),
                 (np.array(cfg['lower1']), np.array(cfg['upper1']),
                  np.array(cfg['lower2']), np.array(cfg['upper2']))))

def hsv_work(frame, arrs):
    l1, u1, l2, u2 = arrs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K, iterations=2),
        cv2.MORPH_CLOSE, _K, iterations=2)
    cv2.connectedComponentsWithStats(mask)

def display_work():
    """GUIスレッドの表示処理相当（4台分のcvtColor+resize）"""
    for _, frame, _ in JOBS:
        small = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

_stop = threading.Event()

def infer_cpu_load():
    """推論スレッドのCPU側負荷（前後処理・NMS相当）を模擬。GPU部分は再現しない。"""
    a = np.random.rand(640, 640).astype(np.float32)
    while not _stop.is_set():
        b = a * 1.0001
        b.sum()
        time.sleep(0.005)

def run_config(name, parallel, cv2_threads):
    cv2.setNumThreads(cv2_threads)
    _stop.clear()
    lat = {cam: [] for cam, _, _ in JOBS}
    peak_threads = [PROC.num_threads()]

    inf = threading.Thread(target=infer_cpu_load, daemon=True)
    inf.start()

    def worker(cam, frame, arrs):
        """カメラスレッド: 20Hzでペーシングしながら HSV を実行"""
        nxt = time.perf_counter()
        while not _stop.is_set():
            t0 = time.perf_counter()
            hsv_work(frame, arrs)
            lat[cam].append((time.perf_counter() - t0) * 1000.0)
            nxt += PERIOD
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()   # 締切超過 → 追いつけていない

    threads = []
    if parallel:
        for cam, frame, arrs in JOBS:
            t = threading.Thread(target=worker, args=(cam, frame, arrs), daemon=True)
            t.start(); threads.append(t)

    # メインスレッド = GUI。表示処理を20Hzで回す
    gui_lat = []
    t_end = time.perf_counter() + DURATION
    nxt = time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        if not parallel:
            # 現状構成: GUIスレッドが4台のHSVを逐次実行
            for cam, frame, arrs in JOBS:
                ts = time.perf_counter()
                hsv_work(frame, arrs)
                lat[cam].append((time.perf_counter() - ts) * 1000.0)
        display_work()
        gui_lat.append((time.perf_counter() - t0) * 1000.0)
        peak_threads.append(PROC.num_threads())
        nxt += PERIOD
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.perf_counter()

    _stop.set()
    for t in threads:
        t.join(timeout=1.0)
    inf.join(timeout=1.0)

    def pct(v, p):
        return statistics.quantiles(v, n=100)[p - 1] if len(v) > 2 else (v[0] if v else 0)

    allc = [x for v in lat.values() for x in v]
    miss = sum(1 for x in gui_lat if x > PERIOD * 1000) / len(gui_lat) * 100
    print(f"  {name}")
    print(f"    OSスレッド数(最大)      : {max(peak_threads)}")
    print(f"    HSV 1台1フレーム p50/p95: {pct(allc,50):6.2f} / {pct(allc,95):6.2f} ms")
    print(f"    GUIティック   p50/p95   : {pct(gui_lat,50):6.2f} / {pct(gui_lat,95):6.2f} ms")
    print(f"    50ms締切超過率          : {miss:5.1f} %")
    print()
    return pct(gui_lat, 95), miss

print(f"=== CPU affinity を 6 コアに制限して計測 (実機は AMD 16コア) ===")
print(f"    各構成 {DURATION}秒 / カメラ20fps / 推論CPU負荷スレッド常駐\n")

print("[A] 現状: GUIスレッドが4台を逐次処理")
run_config("cv2内部スレッド = 6 (既定値)", parallel=False, cv2_threads=6)

print("[B] 4スレッド並列 + cv2内部スレッドも既定のまま  ← ネスト並列の罠")
run_config("cv2内部スレッド = 6 (4スレッド x 6 = 24要求)", parallel=True, cv2_threads=6)

print("[C] 4スレッド並列 + cv2内部スレッドを1に固定  ← 推奨")
run_config("cv2内部スレッド = 1 (4スレッド x 1 = 4要求)", parallel=True, cv2_threads=1)
