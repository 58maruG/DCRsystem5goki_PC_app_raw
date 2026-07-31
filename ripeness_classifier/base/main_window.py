"""メインウィンドウ（本体: 3クラス分類 未熟 / 健全 / 過熟。未熟マスク・過熟マスク占有率モデル）。"""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox,
    QListWidget, QListWidgetItem,
    QGridLayout, QGroupBox, QSplitter,
    QProgressDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor

from ripeness_classifier.base.config import (
    TRAIN_DIR, DEBOUNCE_MS, _DEST_ROOTS,
    DEFAULT_OCC_UNRIPE_CONFIRM, DEFAULT_OCC_OVERRIPE_CONFIRM,
    DEFAULT_MIN_AREA, DEFAULT_STEM_OPEN,
    DEFAULT_REFLECT_SAT_LO, DEFAULT_REFLECT_SAT_HI, DEFAULT_REFLECT_V_LO, DEFAULT_REFLECT_V_HI,
    COLOR_UNRIPE, COLOR_OVERRIPE,
    load_config, save_config,
)
from ripeness_classifier.base.classify import classify_3
from ripeness_classifier.base.workers import ThumbWorker, ScoreWorker, SaveWorker
from ripeness_classifier.base.widgets import (
    HsvSliderGroup, ThumbnailGrid, scroll_wrap, BlockZoomFilter, make_slider_row, make_range_slider_row,
)


# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("果実熟度分類器（3クラス: 未熟 / 健全 / 過熟）")
        self.resize(1500, 950)

        cfg = load_config()
        self._fruit_params:    dict = cfg["fruit"]
        self._unripe_params:   dict = cfg["unripe"]
        self._overripe_params: dict = cfg["overripe"]
        self._occ_unripe_confirm:   int = cfg.get(
            "occ_unripe_confirm", DEFAULT_OCC_UNRIPE_CONFIRM)    # 未熟確定の占有率境界%
        self._occ_overripe_confirm: int = cfg.get(
            "occ_overripe_confirm", DEFAULT_OCC_OVERRIPE_CONFIRM)  # 過熟確定の占有率境界%
        self._min_area:     int  = cfg.get("min_area",   DEFAULT_MIN_AREA)
        self._stem_open:    int  = cfg.get("stem_open",  DEFAULT_STEM_OPEN)   # 果柄除去(開)半径px
        self._reflect_sat_lo: int = cfg.get("reflect_sat_lo", DEFAULT_REFLECT_SAT_LO)  # 虚像除去 彩度範囲下限
        self._reflect_sat_hi: int = cfg.get("reflect_sat_hi", DEFAULT_REFLECT_SAT_HI)  # 虚像除去 彩度範囲上限
        self._reflect_v_lo: int = cfg.get("reflect_v_lo", DEFAULT_REFLECT_V_LO)  # 虚像除去 明度範囲下限(試験導入)
        self._reflect_v_hi: int = cfg.get("reflect_v_hi", DEFAULT_REFLECT_V_HI)  # 虚像除去 明度範囲上限(試験導入)
        self._debug:        bool = False   # デバッグ表示（マスク段階の可視化）。保存しない

        self._current_cls:  str  = ""
        self._path_cache: dict[str, list[str]] = {}                    # 起動時に収集（パスのみ）
        self._cache: dict[str, list[tuple[str, np.ndarray]]] = {}      # クラス選択時に生成
        self._occ_unripes:   list = []
        self._occ_overripes: list = []
        self._fruit_founds: list = []
        self._rects_list:   list = []
        self._worker:       ScoreWorker | None = None
        self._thumb_worker: ThumbWorker | None = None
        self._thumb_dlg:    QProgressDialog | None = None
        self._save_worker:  SaveWorker | None  = None
        self._save_dlg:     QProgressDialog | None = None
        self._save_ctx:     tuple[str, str] = ("", "")  # (target, mode_label)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_score)

        self._build_ui()
        self._load_images()

    # --------------------------------------------------
    # UI 構築
    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── ヘッダー行1: クラス名 + 統計 + 設定保存 ──
        h1 = QHBoxLayout()
        self._cls_label = QLabel("クラス: （未選択）")
        self._cls_label.setFont(QFont("", 13, QFont.Bold))
        h1.addWidget(self._cls_label)

        self._stat_label = QLabel("healthy: -- / unripe: -- / overripe: -- / 未検出: --")
        self._stat_label.setFont(QFont("", 11))
        h1.addWidget(self._stat_label)
        h1.addStretch()

        btn_cfg = QPushButton("設定を保存\n(hsv_ripeness_config.json)")
        btn_cfg.setFixedSize(190, 46)
        btn_cfg.setToolTip("果実マスク・未熟マスク・過熟マスク・占有率境界（未熟/過熟）を JSON に保存します。")
        btn_cfg.clicked.connect(self._save_cfg)
        h1.addWidget(btn_cfg)
        root.addLayout(h1)

        # ── ヘッダー行2: 保存ボタン（3クラス × コピー/クロップ）──
        h2 = QHBoxLayout()
        h2.addStretch()
        save_buttons = [
            ("healthy コピー\n(ripeness_healthy/)",   "healthy",  "copy"),
            ("healthy クロップ\n(ripeness_healthy/)",  "healthy",  "crop"),
            ("unripe コピー\n(ripeness_unripe/)",      "unripe",   "copy"),
            ("unripe クロップ\n(ripeness_unripe/)",     "unripe",   "crop"),
            ("overripe コピー\n(ripeness_overripe/)",  "overripe", "copy"),
            ("overripe クロップ\n(ripeness_overripe/)", "overripe", "crop"),
        ]
        for label, target, mode in save_buttons:
            btn = QPushButton(label)
            btn.setFixedSize(165, 46)
            btn.clicked.connect(
                lambda _=False, t=target, m=mode: self._start_save(t, m)
            )
            h2.addWidget(btn)
        root.addLayout(h2)

        # ── メイン（左パネル ↔ サムネイル領域：横スプリッタで幅をドラッグ調整） ──
        self._main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self._main_splitter, 1)

        # 左パネル：各ブロックを縦スプリッタで高さ配分（境目のハンドルをドラッグ）。
        #   マスク系ブロックはスクロール領域で包み、縮めてもスクロールで全内容に届くようにする。
        self._left_splitter = QSplitter(Qt.Vertical)
        self._left_splitter.setChildrenCollapsible(False)

        # 各ブロックをスクロール領域で包み、Ctrl+ホイールで文字サイズ・幅をズームできるようにする
        def _add_block(block: QWidget, scrollable: bool = True):
            if scrollable:
                sa = scroll_wrap(block)
                BlockZoomFilter(block, sa)          # 幅はスクロール領域→左パネルへ波及
                self._left_splitter.addWidget(sa)
            else:
                BlockZoomFilter(block, block)       # クラス一覧は自身の幅を広げる
                self._left_splitter.addWidget(block)

        _add_block(self._build_procedure_group())

        self._fruit_grp = HsvSliderGroup("果実マスク（黄〜橙〜赤 全体）", self._fruit_params)
        self._fruit_grp.changed.connect(self._on_params_changed)
        _add_block(self._fruit_grp)

        # 未熟色（黄〜緑）は0/180をまたがないので、色相バーは1本だけにする
        self._unripe_grp = HsvSliderGroup(
            "未熟マスク（未熟の色相＋未熟占有率）", self._unripe_params, single_hue=True)
        self._unripe_grp.changed.connect(self._on_params_changed)
        _add_block(self._unripe_grp)

        self._overripe_grp = HsvSliderGroup("過熟マスク（過熟の色相・明度＋過熟占有率）", self._overripe_params)
        self._overripe_grp.changed.connect(self._on_params_changed)
        _add_block(self._overripe_grp)

        _add_block(self._build_detect_group())
        _add_block(self._build_unripe_group())
        _add_block(self._build_overripe_group())
        # クラス一覧はリスト自体がスクロールするので二重スクロールを避けて直接入れる
        _add_block(self._build_class_list(), scrollable=False)
        # 初期の高さ配分（内容量に応じた目安。以後はドラッグで自由に変えられる）
        self._left_splitter.setSizes([120, 230, 230, 230, 170, 80, 80, 220])

        self._main_splitter.addWidget(self._left_splitter)

        self._grid = ThumbnailGrid()
        self._main_splitter.addWidget(self._grid)
        self._main_splitter.setStretchFactor(0, 0)   # 左パネルは固定寄り
        self._main_splitter.setStretchFactor(1, 1)   # サムネイル領域が余りを取る
        self._main_splitter.setSizes([340, 1140])

        self._sb = self.statusBar()
        self._sb.showMessage("画像読み込み中...")

    def _build_procedure_group(self) -> QGroupBox:
        """調整手順（A→B の順序）を常時表示するガイド。"""
        grp = QGroupBox("調整手順（上から順に・A→B の一方向）")
        v = QVBoxLayout(grp)
        v.setContentsMargins(6, 2, 6, 4)
        v.setSpacing(2)
        text = QLabel(
            "<b>A. マスク形状（緑枠）を確定</b><br>"
            "&nbsp;1. 果実マスクHSV（最初に固定・以後触らない）<br>"
            "&nbsp;2. 最小面積 &nbsp; 3. 反射除去 &nbsp; 4. 果柄除去<br>"
            "<b>B. 分類（色ラベル）を確定</b><br>"
            "&nbsp;未熟: 5. 未熟マスクHSV（未熟の色相を兼ねる）<br>"
            "&nbsp;&nbsp;&nbsp;6. 未熟占有率%（以上なら未熟条件を満たす）<br>"
            "&nbsp;過熟: 7. 過熟マスクHSV（過熟の色相・明度を兼ねる）<br>"
            "&nbsp;&nbsp;&nbsp;8. 過熟占有率%（以上なら過熟条件を満たす）<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;（両方の条件を満たしたときは占有率が高い方を採用。"
            "どちらも満たさなければ健全）<br>"
            "<span style='color:gray;'>枠がおかしい→A / 色ラベルがおかしい→B。"
            "Bで困ってもAの色は戻さない。</span>"
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        v.addWidget(text)
        return grp

    def _colored_group(self, title: str, color: QColor | None) -> tuple[QGroupBox, QGridLayout]:
        """判定クラスの色（未熟=橙・過熟=紫・健全=緑）で枠とタイトルを色分けしたグループを作る。
        color=None のときは無色（検出フィルタなど、クラス判定に紐付かないブロック用）。"""
        grp = QGroupBox(title)
        if color is not None:
            rgb = f"rgb({color.red()},{color.green()},{color.blue()})"
            grp.setStyleSheet(
                f"QGroupBox {{ font-weight: bold; color: {rgb}; "
                f"border: 1px solid {rgb}; border-radius: 4px; margin-top: 6px; }} "
                f"QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 3px; }}"
            )
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)      # 行どうしの縦の隙間を最小限に
        grid.setContentsMargins(6, 2, 6, 2)
        grid.setColumnStretch(1, 1)
        return grp, grid

    def _build_detect_group(self) -> QGroupBox:
        """検出フィルタ（マスク形状の整形）。クラス判定そのものには関わらないので無色。"""
        grp, grid = self._colored_group("検出フィルタ（マスク形状）", None)

        # 最小面積
        grid.addWidget(QLabel("最小面積"), 0, 0)
        self._area_slider, self._area_spin = make_slider_row(0, 3000, self._min_area)
        self._area_slider.valueChanged.connect(self._on_area)
        grid.addWidget(self._area_slider, 0, 1)
        grid.addWidget(self._area_spin,   0, 2)

        # 虚像除去（彩度範囲）: HSVマスクと同じ左右ハンドルで、虚像とみなすS範囲を指定する。
        #   典型的には低彩度側（淡い反射）を範囲に含める。全域(0〜255)なら無効。
        #   スライダー順は実際の処理順（虚像除去→果柄除去。compute_ripeness／デバッグ凡例の黄→桃）に合わせる
        grid.addWidget(QLabel("虚像除去\n(彩度S範囲)"), 1, 0)
        self._refs_slider, self._refs_lo_spin, self._refs_hi_spin = make_range_slider_row(
            0, 255, self._reflect_sat_lo, self._reflect_sat_hi)
        self._refs_slider.rangeChanged.connect(self._on_reflect_sat_range)
        grid.addWidget(self._refs_slider,  1, 1)
        grid.addWidget(self._refs_lo_spin, 1, 2)
        grid.addWidget(self._refs_hi_spin, 1, 3)

        # 虚像除去（明度範囲・試験導入）: 虚像とみなすV範囲を指定する。典型的には高明度側（白飛び）を範囲に含める。
        #   彩度範囲と両方有効なときは両条件を同時に満たす画素だけを虚像として除去する
        grid.addWidget(QLabel("虚像除去\n(明度V範囲・試験)"), 2, 0)
        self._refv_slider, self._refv_lo_spin, self._refv_hi_spin = make_range_slider_row(
            0, 255, self._reflect_v_lo, self._reflect_v_hi)
        self._refv_slider.rangeChanged.connect(self._on_reflect_v_range)
        grid.addWidget(self._refv_slider,  2, 1)
        grid.addWidget(self._refv_lo_spin, 2, 2)
        grid.addWidget(self._refv_hi_spin, 2, 3)

        # 果柄除去（開処理半径px）: erode→dilate で細い果柄を切り離す。0=無効、大きいほど太い果柄まで除去
        grid.addWidget(QLabel("果柄除去\n(開処理px)"), 3, 0)
        self._stem_slider, self._stem_spin = make_slider_row(0, 30, self._stem_open)
        self._stem_slider.valueChanged.connect(self._on_stem)
        grid.addWidget(self._stem_slider, 3, 1)
        grid.addWidget(self._stem_spin,   3, 2)

        # デバッグ表示: マスクの段階を色分け輪郭で重ねて、どの段で削れたか可視化する
        self._debug_chk = QCheckBox("デバッグ: マスク段階を表示")
        self._debug_chk.setChecked(self._debug)
        self._debug_chk.setToolTip(
            "サムネイルにマスクの各段階の輪郭を重ねます。\n"
            "白=生マスク / 黄=虚像除去後 / 桃=果柄除去後(最終) / 緑=クロップ枠 / "
            "青=未熟マスク(果実内) / 黒=過熟マスク(果実内)。\n"
            "『白→黄』で縮んだら虚像除去(彩度/明度)、『黄→桃』で縮んだら果柄除去が原因です。\n"
            "青=未熟マスクは未熟の色相・占有率、黒=過熟マスクは過熟の色相・明度・占有率の算出対象です。"
        )
        self._debug_chk.toggled.connect(self._on_debug)
        grid.addWidget(self._debug_chk, 4, 0, 1, 3)

        self._debug_legend = QLabel(
            "<span style='color:white;'>白</span>=生 / "
            "<span style='color:yellow;'>黄</span>=虚像除去後 / "
            "<span style='color:magenta;'>桃</span>=果柄除去後 / "
            "<span style='color:lime;'>緑</span>=クロップ枠 / "
            "<span style='color:#3a8fe0;'>青</span>=未熟マスク(果実内) / "
            "<span style='color:black;'>黒</span>=過熟マスク(果実内)"
        )
        self._debug_legend.setTextFormat(Qt.RichText)
        self._debug_legend.setWordWrap(True)
        grid.addWidget(self._debug_legend, 5, 0, 1, 3)

        return grp

    def _build_unripe_group(self) -> QGroupBox:
        """未熟判定（未熟占有率）。色相は未熟マスクのHSV範囲そのもので定義される。"""
        grp, grid = self._colored_group("未熟判定", COLOR_UNRIPE)

        # 未熟占有率境界: 未熟マスク∩果実マスクの占有率(occ_unripe)がこれ以上なら未熟条件を満たす
        grid.addWidget(QLabel("未熟占有率(%)\n以上なら未熟"), 0, 0)
        self._ouc_slider, self._ouc_spin = make_slider_row(0, 100, self._occ_unripe_confirm, suffix="%")
        self._ouc_slider.valueChanged.connect(self._on_occ_unripe_confirm)
        grid.addWidget(self._ouc_slider, 0, 1)
        grid.addWidget(self._ouc_spin,   0, 2)

        return grp

    def _build_overripe_group(self) -> QGroupBox:
        """過熟判定（過熟占有率）。色相・明度は過熟マスクのHSV範囲そのもので定義される。
        未熟・過熟の両条件を満たす場合は占有率が高い方を採用し、どちらも満たさなければ健全。"""
        grp, grid = self._colored_group("過熟判定", COLOR_OVERRIPE)

        # 過熟占有率境界: 過熟マスク∩果実マスクの占有率(occ_overripe)がこれ以上なら過熟条件を満たす
        grid.addWidget(QLabel("過熟占有率(%)\n以上なら過熟"), 0, 0)
        self._ooc_slider, self._ooc_spin = make_slider_row(0, 100, self._occ_overripe_confirm, suffix="%")
        self._ooc_slider.valueChanged.connect(self._on_occ_overripe_confirm)
        grid.addWidget(self._ooc_slider, 0, 1)
        grid.addWidget(self._ooc_spin,   0, 2)

        return grp

    def _build_class_list(self) -> QGroupBox:
        grp = QGroupBox("クラス一覧（クリックで切り替え）")
        v = QVBoxLayout(grp)
        self._class_list = QListWidget()
        self._class_list.setMinimumWidth(160)   # 幅は横スプリッタで可変（下限のみ）
        self._class_list.setMinimumHeight(80)
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        v.addWidget(self._class_list)
        return grp

    # --------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------
    def _load_images(self):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        train_path = Path(TRAIN_DIR)

        # 直下に画像がある場合はフラット構造としてフォルダ自体を1クラス扱い
        has_direct_images = any(
            f.is_file() and f.suffix.lower() in img_exts
            for f in train_path.iterdir()
        )
        if has_direct_images:
            class_dirs = [(train_path.name, train_path)]
        else:
            class_dirs = [
                (d.name, d) for d in sorted(train_path.iterdir())
                if d.is_dir() and not d.name.startswith("_")
            ]

        # ファイルパスのみ収集（サムネイル生成はクラス選択時に行う）
        for cls_name, class_dir in class_dirs:
            paths = [
                str(fpath) for fpath in sorted(class_dir.iterdir())
                if fpath.is_file() and fpath.suffix.lower() in img_exts
            ]
            if paths:
                self._path_cache[cls_name] = paths

        self._populate_class_list()
        if self._class_list.count() > 0:
            self._class_list.setCurrentRow(0)

    def _populate_class_list(self):
        self._class_list.clear()
        for cls, paths in self._path_cache.items():
            item = QListWidgetItem(f"  {cls}  ({len(paths)}枚)")
            item.setData(Qt.UserRole, cls)
            self._class_list.addItem(item)

    # --------------------------------------------------
    # クラス選択
    # --------------------------------------------------
    def _on_class_selected(self, row: int):
        if row < 0:
            return
        item = self._class_list.item(row)
        cls = item.data(Qt.UserRole)
        if cls == self._current_cls:
            return
        self._current_cls = cls
        self._cls_label.setText(f"クラス: {cls}")

        if cls in self._cache:
            # サムネイル生成済みならすぐ表示
            items = self._cache[cls]
            self._reset_result_arrays(len(items))
            self._push_debug_context()
            self._grid.update_grid(
                items, self._occ_unripes, self._occ_overripes,
                self._fruit_founds, self._rects_list,
                self._occ_unripe_confirm / 100.0,
                self._occ_overripe_confirm / 100.0
            )
            self._run_score()
        else:
            # 未生成ならバックグラウンドで生成
            self._grid.clear_grid()
            self._reset_result_arrays(0)
            self._stat_label.setText("読み込み中...")
            paths = self._path_cache.get(cls, [])
            self._start_thumb_loading(cls, paths)

    def _reset_result_arrays(self, n: int):
        self._occ_unripes   = [0.0]   * n
        self._occ_overripes = [0.0]   * n
        self._fruit_founds  = [False] * n
        self._rects_list    = [[]]    * n

    def _start_thumb_loading(self, cls_name: str, paths: list):
        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.quit()
            self._thumb_worker.wait()
        if self._thumb_dlg:
            self._thumb_dlg.close()

        n = len(paths)
        self._sb.showMessage(f"サムネイル生成中... ({n} 枚)")
        self._thumb_dlg = QProgressDialog(
            f"画像を読み込んでいます... ({n} 枚)", None, 0, n, self
        )
        self._thumb_dlg.setWindowTitle("読み込み中")
        self._thumb_dlg.setWindowModality(Qt.WindowModal)
        self._thumb_dlg.setMinimumDuration(500)
        self._thumb_dlg.setValue(0)

        self._thumb_worker = ThumbWorker(paths)
        self._thumb_worker.progress.connect(self._thumb_dlg.setValue)
        self._thumb_worker.finished.connect(
            lambda entries: self._on_thumb_done(cls_name, entries)
        )
        self._thumb_worker.start()

    def _on_thumb_done(self, cls_name: str, entries: list):
        if self._thumb_dlg:
            self._thumb_dlg.close()
            self._thumb_dlg = None

        self._cache[cls_name] = entries

        if cls_name == self._current_cls:
            self._reset_result_arrays(len(entries))
            self._push_debug_context()
            self._grid.update_grid(
                entries, self._occ_unripes, self._occ_overripes,
                self._fruit_founds, self._rects_list,
                self._occ_unripe_confirm / 100.0,
                self._occ_overripe_confirm / 100.0
            )
            self._run_score()

        self._sb.showMessage(f"読み込み完了: {len(entries)} 枚")

    # --------------------------------------------------
    # スライダー操作
    # --------------------------------------------------
    def _on_params_changed(self):
        self._fruit_params    = self._fruit_grp.get_params()
        self._unripe_params   = self._unripe_grp.get_params()
        self._overripe_params = self._overripe_grp.get_params()
        self._debounce.start(DEBOUNCE_MS)

    def _on_area(self, val: int):
        self._min_area = val
        self._debounce.start(DEBOUNCE_MS)

    def _on_stem(self, val: int):
        # 果柄除去はマスク形状（輪郭・bbox・占有率）を変えるので再計算が必要
        self._stem_open = val
        self._debounce.start(DEBOUNCE_MS)

    def _on_reflect_sat_range(self, lo: int, hi: int):
        # 虚像除去（彩度）もマスク形状を変えるので再計算が必要
        self._reflect_sat_lo, self._reflect_sat_hi = lo, hi
        self._debounce.start(DEBOUNCE_MS)

    def _on_reflect_v_range(self, lo: int, hi: int):
        # 虚像除去（明度）もマスク形状を変えるので再計算が必要
        self._reflect_v_lo, self._reflect_v_hi = lo, hi
        self._debounce.start(DEBOUNCE_MS)

    def _on_debug(self, checked: bool):
        # デバッグ表示は描画だけの切り替え。スコア再計算は不要で再描画のみ
        self._debug = checked
        self._refresh_grid()

    def _on_occ_unripe_confirm(self, val: int):
        # occ_unripe は計算済み。未熟の占有率境界だけ動かして色判定を更新する（再計算不要）
        self._occ_unripe_confirm = val
        self._refresh_grid()

    def _on_occ_overripe_confirm(self, val: int):
        # occ_overripe は計算済み。過熟の占有率境界だけ動かして色判定を更新する（再計算不要）
        self._occ_overripe_confirm = val
        self._refresh_grid()

    # --------------------------------------------------
    # スコア計算（バックグラウンド）
    # --------------------------------------------------
    def _run_score(self):
        if not self._current_cls:
            return
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        items = self._cache.get(self._current_cls, [])
        if not items:
            return
        self._sb.showMessage("処理中...")
        self._worker = ScoreWorker(
            items, self._fruit_params, self._unripe_params, self._overripe_params,
            self._min_area, self._stem_open,
            self._reflect_sat_lo, self._reflect_sat_hi, self._reflect_v_lo, self._reflect_v_hi
        )
        self._worker.finished.connect(self._on_score_done)
        self._worker.start()

    def _on_score_done(self, occ_unripes: list, occ_overripes: list,
                       fruit_founds: list, rects_list: list):
        self._occ_unripes   = occ_unripes
        self._occ_overripes = occ_overripes
        self._fruit_founds  = fruit_founds
        self._rects_list    = rects_list
        self._refresh_grid()
        self._sb.showMessage("完了")

    # --------------------------------------------------
    # 表示更新
    # --------------------------------------------------
    def _push_debug_context(self):
        # デバッグ描画に必要な現在のマスクパラメータをグリッドへ渡す
        self._grid.set_debug_context(
            self._debug, self._fruit_params, self._unripe_params, self._overripe_params,
            self._stem_open,
            self._reflect_sat_lo, self._reflect_sat_hi, self._reflect_v_lo, self._reflect_v_hi
        )

    def _refresh_grid(self):
        if not self._current_cls:
            return
        items = self._cache.get(self._current_cls, [])
        occ_un = self._occ_unripe_confirm / 100.0
        occ_ov = self._occ_overripe_confirm / 100.0
        self._push_debug_context()
        self._grid.update_grid(
            items, self._occ_unripes, self._occ_overripes,
            self._fruit_founds, self._rects_list,
            occ_un, occ_ov
        )
        self._update_stat(occ_un, occ_ov)

    def _update_stat(self, occ_un: float, occ_ov: float):
        counts = {"healthy": 0, "unripe": 0, "overripe": 0, "none": 0}
        for ou, oo, ff in zip(self._occ_unripes, self._occ_overripes, self._fruit_founds):
            counts[classify_3(ou, oo, ff, occ_un, occ_ov)] += 1
        self._stat_label.setText(
            f"<span style='color:lime;font-weight:bold;'>healthy: {counts['healthy']}</span>  /  "
            f"<span style='color:orange;font-weight:bold;'>unripe: {counts['unripe']}</span>  /  "
            f"<span style='color:violet;font-weight:bold;'>overripe: {counts['overripe']}</span>  /  "
            f"<span style='color:tomato;'>未検出: {counts['none']}</span>"
        )

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    def _save_cfg(self):
        save_config({
            "fruit":      self._fruit_params,
            "unripe":     self._unripe_params,
            "overripe":   self._overripe_params,
            "occ_unripe_confirm":   self._occ_unripe_confirm,
            "occ_overripe_confirm": self._occ_overripe_confirm,
            "min_area":     self._min_area,
            "stem_open":    self._stem_open,
            "reflect_sat_lo": self._reflect_sat_lo,
            "reflect_sat_hi": self._reflect_sat_hi,
            "reflect_v_lo":   self._reflect_v_lo,
            "reflect_v_hi":   self._reflect_v_hi,
        })
        self._sb.showMessage("設定を保存しました: json/hsv_ripeness_config.json")

    def _start_save(self, target: str, save_mode: str):
        if not self._current_cls:
            QMessageBox.warning(self, "警告", "クラスを選択してください。")
            return
        # SaveWorker はパスのみ使うので _path_cache から取得（サムネイル未生成でも可）
        paths = self._path_cache.get(self._current_cls, [])
        if not paths:
            QMessageBox.information(self, "情報", "このクラスに画像がありません。")
            return

        mode_label = "コピー" if save_mode == "copy" else "クロップ"
        occ_un = self._occ_unripe_confirm / 100.0
        occ_ov = self._occ_overripe_confirm / 100.0
        # プレビュー枚数（サムネイル基準。保存も同じ基準で再判定するので枚数は一致する）
        count = sum(
            1 for ou, oo, ff in zip(self._occ_unripes, self._occ_overripes, self._fruit_founds)
            if classify_3(ou, oo, ff, occ_un, occ_ov) == target
        )
        dest = _DEST_ROOTS[target] / self._current_cls

        reply = QMessageBox.question(
            self, "確認",
            f"クラス「{self._current_cls}」のうち {target} と判定された\n"
            f"約 {count} 枚を\n{dest}\nに{mode_label}保存します。よろしいですか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_ctx = (target, mode_label)
        self._save_dlg = QProgressDialog(
            f"{target} 画像を{mode_label}保存中...", None, 0, len(paths), self
        )
        self._save_dlg.setWindowTitle(f"{mode_label}保存中")
        self._save_dlg.setWindowModality(Qt.WindowModal)
        self._save_dlg.setMinimumDuration(0)
        self._save_dlg.setValue(0)
        self._save_dlg.show()

        self._save_worker = SaveWorker(
            self._current_cls, paths,
            self._fruit_params, self._unripe_params, self._overripe_params,
            self._min_area, self._stem_open,
            self._reflect_sat_lo, self._reflect_sat_hi, self._reflect_v_lo, self._reflect_v_hi,
            occ_un, occ_ov,
            target, save_mode
        )
        self._save_worker.progress.connect(self._save_dlg.setValue)
        self._save_worker.finished.connect(self._on_save_done)
        self._save_worker.start()

    def _on_save_done(self, saved: int, total: int):
        if self._save_dlg:
            self._save_dlg.close()
            self._save_dlg = None

        target, mode_label = self._save_ctx
        dest = _DEST_ROOTS[target] / self._current_cls

        QMessageBox.information(
            self, "保存完了",
            f"{total} 枚を処理し、{saved} 件を{mode_label}しました。\n"
            f"種別: {target}\n"
            f"保存先: {dest}"
        )
        self._sb.showMessage(f"{target} {mode_label}完了: {saved}/{total} 件")
