"""ホットパス（HSV判定・色変換・Qtスケーリング）の実測ベンチ。
実際の pfs 解像度と hsv_config_*.json を使う。"""
import json, os, pathlib, time
import numpy as np
import cv2

# リポジトリルート（このファイルの1つ上）を基準にする
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

# pfs から読み取った実解像度
CAMS = [("cam_top", 640), ("cam_under", 640), ("cam_inside", 560), ("cam_outside", 500)]
N = 200

# サクランボを模した赤い円を置いた擬似フレーム（HSVマスクが実際にヒットするように）
def make_frame(size):
    f = np.random.randint(0, 60, (size, size, 3), dtype=np.uint8)
    cv2.circle(f, (size // 2, size // 2), size // 6, (40, 40, 220), -1)
    return f

def hsv_pipeline(frame, cfg):
    """get_target_info の中身をそのまま再現"""
    lower1, upper1 = np.array(cfg['lower1']), np.array(cfg['upper1'])
    lower2, upper2 = np.array(cfg['lower2']), np.array(cfg['upper2'])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2),
        cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.connectedComponentsWithStats(mask)

def bench(fn, n=N):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0

print(f"cv2.getNumThreads() = {cv2.getNumThreads()}")
print()

total_hsv = 0.0
for cam, size in CAMS:
    with open(f"json/hsv_config_{cam}.json") as f:
        cfg = json.load(f)
    frame = make_frame(size)

    ms_all = bench(lambda: hsv_pipeline(frame, cfg))
    total_hsv += ms_all

    # 内訳
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l1, u1 = np.array(cfg['lower1']), np.array(cfg['upper1'])
    l2, u2 = np.array(cfg['lower2']), np.array(cfg['upper2'])
    mask = cv2.bitwise_or(cv2.inRange(hsv_img, l1, u1), cv2.inRange(hsv_img, l2, u2))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ms_cvt = bench(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    ms_morph = bench(lambda: cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=2),
        cv2.MORPH_CLOSE, kern, iterations=2))
    ms_cc = bench(lambda: cv2.connectedComponentsWithStats(mask))
    ms_rgb = bench(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ms_resize = bench(lambda: cv2.resize(frame, (640, 640)))

    print(f"{cam} ({size}x{size}):")
    print(f"   HSV全体            {ms_all:6.2f} ms")
    print(f"     - cvtColor HSV   {ms_cvt:6.2f} ms")
    print(f"     - morphology x4  {ms_morph:6.2f} ms  <-- OPEN2回+CLOSE2回")
    print(f"     - connectedComp  {ms_cc:6.2f} ms  (labels {size*size*4/1e6:.2f}MB を毎回確保)")
    print(f"   cvtColor BGR2RGB   {ms_rgb:6.2f} ms")
    print(f"   resize->640        {ms_resize:6.2f} ms")
    print()

print(f"=== HSV 4台合計: {total_hsv:.2f} ms / QTimerティック(50ms周期) ===")
print(f"    50ms 予算に対する占有率: {total_hsv/50*100:.1f}%")
