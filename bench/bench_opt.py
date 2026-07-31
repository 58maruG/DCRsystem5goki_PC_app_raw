"""スレッド数制約時のHSVコストと、最適化案の効果を実測。"""
import json, os, pathlib, time
import numpy as np
import cv2

# リポジトリルート（このファイルの1つ上）を基準にする
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
CAMS = [("cam_top", 640), ("cam_under", 640), ("cam_inside", 560), ("cam_outside", 500)]
N = 200

def make_frame(size):
    f = np.random.randint(0, 60, (size, size, 3), dtype=np.uint8)
    cv2.circle(f, (size // 2, size // 2), size // 6, (40, 40, 220), -1)
    return f

def bench(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0

# --- 現状実装 ---
def current(frame, cfg):
    l1, u1 = np.array(cfg['lower1']), np.array(cfg['upper1'])
    l2, u2 = np.array(cfg['lower2']), np.array(cfg['upper2'])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2),
        cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.connectedComponentsWithStats(mask)

# --- 最適化案: 配列/カーネル事前生成 + morphology 1回 + CCL stats のみ ---
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
def optimized(frame, arrs):
    l1, u1, l2, u2 = arrs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K, iterations=1),
        cv2.MORPH_CLOSE, _K, iterations=1)
    return cv2.connectedComponentsWithStats(mask)

# --- 半分解像度でHSV判定（YOLO入力は原寸のまま） ---
def half_res(frame, arrs):
    l1, u1, l2, u2 = arrs
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                       interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K, iterations=1),
        cv2.MORPH_CLOSE, _K, iterations=1)
    return cv2.connectedComponentsWithStats(mask)

for nthreads in (16, 6, 2, 1):
    cv2.setNumThreads(nthreads)
    tot_cur = tot_opt = tot_half = 0.0
    for cam, size in CAMS:
        with open(f"json/hsv_config_{cam}.json") as f:
            cfg = json.load(f)
        arrs = (np.array(cfg['lower1']), np.array(cfg['upper1']),
                np.array(cfg['lower2']), np.array(cfg['upper2']))
        frame = make_frame(size)
        tot_cur += bench(lambda: current(frame, cfg))
        tot_opt += bench(lambda: optimized(frame, arrs))
        tot_half += bench(lambda: half_res(frame, arrs))
    print(f"cv2 threads={nthreads:2d}  現状 {tot_cur:6.2f}ms | 最適化 {tot_opt:6.2f}ms "
          f"| 半解像度 {tot_half:6.2f}ms   (50ms予算比 現状{tot_cur/50*100:5.1f}% -> 半解像度{tot_half/50*100:4.1f}%)")
