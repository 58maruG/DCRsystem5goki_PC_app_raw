"""4カメラのHSV処理を「逐次」vs「4スレッド並列」で比較。
GIL がボトルネックになるか（=OpenCVがGILを解放するか）を実測で確かめる。
Jetson想定に近づけるため cv2 の内部スレッドは 1 に固定する。"""
import json, os, pathlib, time, threading
import numpy as np, cv2

# リポジトリルート（このファイルの1つ上）を基準にする
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
cv2.setNumThreads(1)   # 内部並列を切り、純粋にPythonスレッドの効果を見る

CAMS = [("cam_top", 640), ("cam_under", 640), ("cam_inside", 560), ("cam_outside", 500)]
N = 100
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def make_frame(size):
    f = np.random.randint(0, 60, (size, size, 3), dtype=np.uint8)
    cv2.circle(f, (size // 2, size // 2), size // 6, (40, 40, 220), -1)
    return f

def work(frame, arrs):
    l1, u1, l2, u2 = arrs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K, iterations=2),
        cv2.MORPH_CLOSE, _K, iterations=2)
    cv2.connectedComponentsWithStats(mask)

jobs = []
for cam, size in CAMS:
    with open(f"json/hsv_config_{cam}.json") as f:
        cfg = json.load(f)
    jobs.append((make_frame(size),
                 (np.array(cfg['lower1']), np.array(cfg['upper1']),
                  np.array(cfg['lower2']), np.array(cfg['upper2']))))

# --- 逐次（現状: GUIスレッドが4台をforループ） ---
def sequential():
    for frame, arrs in jobs:
        work(frame, arrs)

# --- 4スレッド並列（提案: カメラごとのスレッドで処理） ---
def parallel():
    ts = [threading.Thread(target=work, args=(f, a)) for f, a in jobs]
    for t in ts: t.start()
    for t in ts: t.join()

def bench(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0

seq = bench(sequential)
par = bench(parallel)
print(f"逐次 (現状のGUIスレッドforループ) : {seq:6.2f} ms")
print(f"4スレッド並列                      : {par:6.2f} ms")
print(f"高速化率                           : {seq/par:.2f}x")
print()
print("→ 1.0x 近辺なら GIL で詰まっている / 2x以上なら OpenCV が GIL を解放し並列化が効く")
