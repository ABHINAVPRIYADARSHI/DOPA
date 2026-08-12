"""
DOPA - Desktop Occlusal Plane Analyzer
Rewritten with:
  - PyQt5 main window visible immediately on startup
  - Background mesh loading via QThread (no UI freeze)
  - KDTree-based O(n log n) cusp detection (replaces O(n²) loop)
  - Animated spinner during processing
"""

import sys
import os
import threading

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import pyvista as pv
from pyvistaqt import BackgroundPlotter   # embeds VTK inside Qt window

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QProgressBar,
    QSizePolicy, QFrame, QSplitter, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt5.QtGui import QFont, QIcon, QMovie, QPixmap

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MONSON_RADIUS   = 101.6   # mm
CUSP_RADIUS     = 2.0     # local neighbourhood radius for local-max search
DBSCAN_EPS      = 3.0     # mm  – merge nearby cusp candidates
DBSCAN_MIN_PTS  = 1

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MESH = os.path.join(APP_DIR, "mandible.obj.obj")


# ---------------------------------------------------------------------------
# CUSP DETECTION  –  O(n log n) via cKDTree + DBSCAN cluster pruning
# ---------------------------------------------------------------------------
def detect_cusps_kdtree(points: np.ndarray,
                         radius: float = CUSP_RADIUS,
                         z_percentile: float = 85,
                         dbscan_eps: float = DBSCAN_EPS) -> np.ndarray:
    """
    1. Keep only the top z_percentile of points  (occlusal surface)
    2. Build a cKDTree over those points
    3. For each point, query neighbours within `radius` – O(n log n) total
    4. A point is a local maximum if its Z is the max in its neighbourhood
    5. DBSCAN merges nearby local maxima → one representative per cusp
    Returns array of shape (N, 3).
    """
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, z_percentile)
    occ = points[z_vals > z_thresh]

    if len(occ) == 0:
        return np.empty((0, 3))

    tree = cKDTree(occ)

    # Query all neighbours at once – returns list of index arrays
    neighbour_lists = tree.query_ball_point(occ, r=radius)

    # Vectorised local-max check
    local_max_mask = np.array([
        occ[i, 2] >= np.max(occ[idxs, 2])
        for i, idxs in enumerate(neighbour_lists)
    ], dtype=bool)

    candidates = occ[local_max_mask]

    if len(candidates) == 0:
        return candidates

    # DBSCAN: merge candidates that are within dbscan_eps of each other
    labels = DBSCAN(eps=dbscan_eps, min_samples=DBSCAN_MIN_PTS,
                    algorithm='ball_tree').fit_predict(candidates)

    cusps = []
    for lbl in set(labels):
        cluster = candidates[labels == lbl]
        # Pick the highest point in each cluster as the cusp tip
        cusps.append(cluster[np.argmax(cluster[:, 2])])

    return np.array(cusps)


# ---------------------------------------------------------------------------
# MONSON SPHERE SOLVER
# ---------------------------------------------------------------------------
def compute_monson(three_points: np.ndarray) -> np.ndarray:
    p1, p2, p3 = three_points
    R = MONSON_RADIUS

    def equations(center):
        return [
            np.linalg.norm(p1 - center) - R,
            np.linalg.norm(p2 - center) - R,
            np.linalg.norm(p3 - center) - R,
        ]

    center_guess = np.mean(three_points, axis=0)
    return np.array(fsolve(equations, center_guess))


# ---------------------------------------------------------------------------
# BACKGROUND WORKER THREAD
# ---------------------------------------------------------------------------
class MeshLoader(QThread):
    """Loads mesh and detects cusps in a background thread."""
    progress_msg  = pyqtSignal(str)          # status text updates
    finished_ok   = pyqtSignal(object, object)  # (mesh, cusp_points)
    finished_err  = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.progress_msg.emit("Loading mesh from disk …")
            mesh = pv.read(self.path)
            points = mesh.points

            self.progress_msg.emit(
                f"Mesh loaded — {len(points):,} vertices. Detecting cusps …"
            )

            cusps = detect_cusps_kdtree(points)

            self.progress_msg.emit(
                f"Done — {len(cusps)} cusp tip(s) detected."
            )
            self.finished_ok.emit(mesh, cusps)

        except Exception as exc:
            self.finished_err.emit(str(exc))


# ---------------------------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------------------------
class DOPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SanVie")
        self.resize(1280, 800)
        self._set_icon()

        self.mesh       = None
        self.cusp_pts   = None
        self.selected   = []
        self.loader     = None
        self.plotter    = None
        self._spinner_angle = 0

        self._build_ui()
        self._start_spinner()

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------
    def _set_icon(self):
        ico = os.path.join(APP_DIR, "logo.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QIcon(ico))

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Top toolbar ────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setFixedHeight(56)
        toolbar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f172a, stop:1 #1e293b);
                border-bottom: 1px solid #334155;
            }
        """)
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(18, 0, 18, 0)

        title_lbl = QLabel("SanVie 🦷  Digital Occlusal Plane Analyser")
        title_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title_lbl.setStyleSheet("color: #f8fafc; background: transparent;")
        tb_lay.addWidget(title_lbl)

        tb_lay.addStretch()

        btn_style_primary = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0284c7, stop:1 #0369a1);
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                padding: 0 16px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover   { background: #38bdf8; color: #0f172a; border-color: #7dd3fc; }
            QPushButton:pressed { background: #0284c7; color: #ffffff; }
            QPushButton:disabled{ background: #1e293b; color: #475569; border-color: #334155; }
        """
        btn_style_secondary = """
            QPushButton {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 0 16px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover   { background: #334155; color: #ffffff; border-color: #475569; }
            QPushButton:pressed { background: #0f172a; }
            QPushButton:disabled{ background: #0f172a; color: #475569; border-color: #1e293b; }
        """

        self.open_btn = QPushButton("📂  Open Mesh")
        self.open_btn.setFixedHeight(34)
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.setStyleSheet(btn_style_primary)
        self.open_btn.clicked.connect(self._open_file_dialog)
        tb_lay.addWidget(self.open_btn)

        self.reset_btn = QPushButton("🔄  Reset")
        self.reset_btn.setFixedHeight(34)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(btn_style_primary)
        self.reset_btn.clicked.connect(self._reset)
        tb_lay.addWidget(self.reset_btn)

        root_layout.addWidget(toolbar)

        # ── Content Container (Left View Bar + Splitter) ────────────────
        content_widget = QWidget()
        content_lay = QHBoxLayout(content_widget)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(0)

        # ── View presets bar (Vertical Left Sidebar) ───────────────────
        view_bar = QFrame()
        view_bar.setFixedWidth(96)
        view_bar.setStyleSheet("""
            QFrame {
                background: #0f172a;
                border-right: 1px solid #1e293b;
            }
        """)
        vb_lay = QVBoxLayout(view_bar)
        vb_lay.setContentsMargins(8, 14, 8, 14)
        vb_lay.setSpacing(8)

        view_lbl = QLabel("VIEWS")
        view_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        view_lbl.setStyleSheet("color: #64748b; letter-spacing: 1px; background: transparent;")
        vb_lay.addWidget(view_lbl, alignment=Qt.AlignCenter)

        view_btn_style = """
            QPushButton {
                background: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 0px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover   { background: #0284c7; color: #ffffff; border-color: #38bdf8; }
            QPushButton:pressed { background: #0369a1; }
            QPushButton:disabled{ background: #0f172a; color: #475569; border-color: #1e293b; }
        """

        self._view_buttons = []
        for label, view, tooltip in [
            ("↑ Front",  "front",  "Front Camera View"),
            ("↓ Back",   "back",   "Back Camera View"),
            ("▲ Top",    "top",    "Top Occlusal View"),
            ("▼ Bottom", "bottom", "Bottom View"),
            ("◀ Left",   "left",   "Left Sagittal View"),
            ("▶ Right",  "right",  "Right Sagittal View"),
            ("⤢ ISO",    "iso",    "Isometric Perspective"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setToolTip(tooltip)
            btn.setEnabled(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(view_btn_style)
            btn.clicked.connect(lambda checked, v=view: self._set_view(v))
            vb_lay.addWidget(btn)
            self._view_buttons.append(btn)

        vb_lay.addStretch()
        content_lay.addWidget(view_bar)

        # ── Splitter: VTK view  |  Info panel ─────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #1e293b; }")

        # VTK / overlay area
        self.vtk_container = QWidget()
        self.vtk_container.setMinimumWidth(600)
        self.vtk_container.setStyleSheet("background: #090d16;")
        vtk_lay = QVBoxLayout(self.vtk_container)
        vtk_lay.setContentsMargins(0, 0, 0, 0)

        # Placeholder shown before mesh is loaded
        self.placeholder = QWidget()
        self.placeholder.setStyleSheet("background: #090d16;")
        ph_lay = QVBoxLayout(self.placeholder)
        ph_lay.setAlignment(Qt.AlignCenter)
        ph_lay.setSpacing(20)

        ph_icon = QLabel("🦷")
        ph_icon.setAlignment(Qt.AlignCenter)
        ph_icon.setFont(QFont("Segoe UI", 56))
        ph_icon.setStyleSheet("background: transparent;")
        ph_lay.addWidget(ph_icon)

        ph_text = QLabel("No 3D Mesh Loaded")
        ph_text.setAlignment(Qt.AlignCenter)
        ph_text.setFont(QFont("Segoe UI", 16, QFont.Bold))
        ph_text.setStyleSheet("color: #f8fafc; background: transparent;")
        ph_lay.addWidget(ph_text)

        ph_sub = QLabel("Supported Formats: OBJ · STL · PLY · VTK · VTP")
        ph_sub.setAlignment(Qt.AlignCenter)
        ph_sub.setFont(QFont("Segoe UI", 10))
        ph_sub.setStyleSheet("color: #64748b; background: transparent;")
        ph_lay.addWidget(ph_sub)

        self.ph_open_btn = QPushButton("📂   Open Mesh File")
        self.ph_open_btn.setFixedSize(210, 44)
        self.ph_open_btn.setCursor(Qt.PointingHandCursor)
        self.ph_open_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.ph_open_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0284c7, stop:1 #0369a1);
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 8px;
            }
            QPushButton:hover  { background: #38bdf8; color: #0f172a; border-color: #7dd3fc; }
            QPushButton:pressed{ background: #0284c7; color: #ffffff; }
        """)
        self.ph_open_btn.clicked.connect(self._open_file_dialog)
        ph_lay.addWidget(self.ph_open_btn, alignment=Qt.AlignCenter)

        vtk_lay.addWidget(self.placeholder)

        # Processing overlay (hidden initially)
        self.processing_overlay = QWidget(self.vtk_container)
        self.processing_overlay.setStyleSheet(
            "background: rgba(9, 13, 22, 220); border-radius: 12px;"
        )
        overlay_lay = QVBoxLayout(self.processing_overlay)
        overlay_lay.setAlignment(Qt.AlignCenter)

        self.spinner_lbl = QLabel("⏳")
        self.spinner_lbl.setAlignment(Qt.AlignCenter)
        self.spinner_lbl.setFont(QFont("Segoe UI", 44))
        overlay_lay.addWidget(self.spinner_lbl)

        self.proc_lbl = QLabel("Initialising …")
        self.proc_lbl.setAlignment(Qt.AlignCenter)
        self.proc_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.proc_lbl.setStyleSheet("color: #f8fafc; background: transparent;")
        overlay_lay.addWidget(self.proc_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)          # indeterminate
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background: #1e293b;
            }
            QProgressBar::chunk {
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0ea5e9, stop:1 #38bdf8);
            }
        """)
        overlay_lay.addWidget(self.progress_bar, alignment=Qt.AlignCenter)
        self.processing_overlay.hide()

        splitter.addWidget(self.vtk_container)

        # ── Info/results panel ──────────────────────────────────────────
        info_panel = QFrame()
        info_panel.setFixedWidth(290)
        info_panel.setStyleSheet("""
            QFrame {
                background: #0f172a;
                border-left: 1px solid #1e293b;
            }
        """)
        info_lay = QVBoxLayout(info_panel)
        info_lay.setContentsMargins(14, 14, 14, 14)
        info_lay.setSpacing(10)

        card_style = """
            QFrame {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
            }
        """

        # ── Card 1: Detected Cusps ──────────────────
        cusp_card = QFrame()
        cusp_card.setStyleSheet(card_style)
        cc_lay = QVBoxLayout(cusp_card)
        cc_lay.setContentsMargins(12, 10, 12, 10)
        cc_lay.setSpacing(4)
        cc_lay.addWidget(self._section("DETECTED CUSPS"))
        self.cusp_count_lbl = QLabel("—")
        self.cusp_count_lbl.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.cusp_count_lbl.setStyleSheet("color: #34d399; background: transparent;")
        cc_lay.addWidget(self.cusp_count_lbl)
        info_lay.addWidget(cusp_card)

        # ── Card 2: Selection & Monson Sphere ───────
        monson_card = QFrame()
        monson_card.setStyleSheet(card_style)
        mc_lay = QVBoxLayout(monson_card)
        mc_lay.setContentsMargins(12, 10, 12, 10)
        mc_lay.setSpacing(6)

        mc_lay.addWidget(self._section("SELECTION STATUS"))
        self.sel_lbl = QLabel("0 / 3 cusps selected")
        self.sel_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.sel_lbl.setStyleSheet("color: #fbbf24; background: transparent;")
        mc_lay.addWidget(self.sel_lbl)

        mc_lay.addWidget(self._section("MONSON SPHERE"))
        self.monson_lbl = QLabel(
            "Select 3 cusp tips in 3D view\n"
            "(hover over cusp + press P key)\n"
            "to compute Monson sphere."
        )
        self.monson_lbl.setWordWrap(True)
        self.monson_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")
        self.monson_lbl.setFont(QFont("Segoe UI", 10))
        mc_lay.addWidget(self.monson_lbl)
        info_lay.addWidget(monson_card)

        # ── Card 3: Deviation Report ────────────────
        dev_card = QFrame()
        dev_card.setStyleSheet(card_style)
        dc_lay = QVBoxLayout(dev_card)
        dc_lay.setContentsMargins(12, 10, 12, 10)
        dc_lay.setSpacing(6)
        dc_lay.addWidget(self._section("DEVIATION REPORT"))

        dev_scroll = QScrollArea()
        dev_scroll.setWidgetResizable(True)
        dev_scroll.setFrameShape(QFrame.NoFrame)
        dev_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #0f172a;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.dev_lbl = QLabel("—")
        self.dev_lbl.setWordWrap(True)
        self.dev_lbl.setStyleSheet("color: #f8fafc; font-size: 11px; background: transparent;")
        self.dev_lbl.setFont(QFont("Segoe UI", 10))
        self.dev_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        dev_scroll.setWidget(self.dev_lbl)
        dc_lay.addWidget(dev_scroll)
        info_lay.addWidget(dev_card, 1)

        # ── Card 4: Instructions / Legend ───────────
        hint_card = QFrame()
        hint_card.setStyleSheet(card_style)
        hc_lay = QVBoxLayout(hint_card)
        hc_lay.setContentsMargins(12, 10, 12, 10)
        hc_lay.setSpacing(4)
        hc_lay.addWidget(self._section("LEGEND & CONTROLS"))

        hint = QLabel(
            "🟢 Green = Auto-detected cusp\n"
            "🟡 Yellow = User-selected cusp\n"
            "🔵 Blue = Below Monson sphere\n"
            "🔴 Red = Above Monson sphere\n\n"
            "🖱 Left Click + Drag = Rotate\n"
            "🖱 Mouse Scroll = Zoom\n"
            "🖱 Shift + Drag = Pan\n"
            "⌨ Right Click / P Key = Pick Cusp"
        )
        hint.setWordWrap(True)
        hint.setFont(QFont("Segoe UI", 9))
        hint.setStyleSheet("color: #94a3b8; line-height: 1.3; background: transparent;")
        hc_lay.addWidget(hint)
        info_lay.addWidget(hint_card)

        splitter.addWidget(info_panel)
        content_lay.addWidget(splitter, 1)
        root_layout.addWidget(content_widget, 1)

        # ── Status bar ─────────────────────────────────────────────────
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #0f172a;
                color: #94a3b8;
                font-size: 11px;
                border-top: 1px solid #1e293b;
                padding-left: 8px;
            }
        """)
        self.status_bar.showMessage("Ready — open a mesh file to begin.")

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #38bdf8; font-size: 10px; font-weight: bold; "
            "letter-spacing: 1px; background: transparent;"
        )
        lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        return lbl

    # ------------------------------------------------------------------
    # Spinner animation (rotates emoji via timer)
    # ------------------------------------------------------------------
    def _start_spinner(self):
        self._spinner_frames = ["⏳", "⌛", "⏳", "⌛"]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_timer.setInterval(400)

    def _tick_spinner(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.spinner_lbl.setText(self._spinner_frames[self._spinner_idx])

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    def _show_overlay(self, msg: str = "Processing …"):
        self.proc_lbl.setText(msg)
        # Position overlay centred over the vtk container
        self.processing_overlay.setGeometry(self.vtk_container.rect())
        self.processing_overlay.show()
        self.processing_overlay.raise_()
        self._spinner_timer.start()

    def _hide_overlay(self):
        self._spinner_timer.stop()
        self.processing_overlay.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.processing_overlay.isVisible():
            self.processing_overlay.setGeometry(self.vtk_container.rect())

    # ------------------------------------------------------------------
    # File open
    # ------------------------------------------------------------------
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Mesh File",
            APP_DIR,
            "3D Mesh files (*.obj *.stl *.ply *.vtk *.vtp);;"
            "OBJ files (*.obj);;"
            "STL files (*.stl);;"
            "All files (*.*)"
        )
        if not path:
            return
        self._load_mesh(path)

    # ------------------------------------------------------------------
    # Mesh loading pipeline
    # ------------------------------------------------------------------
    def _load_mesh(self, path: str):
        self.open_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        for b in self._view_buttons:
            b.setEnabled(False)
        self.status_bar.showMessage(f"Loading: {os.path.basename(path)} …")
        self._show_overlay("Reading mesh and detecting cusp tips …")

        self.loader = MeshLoader(path, parent=self)
        self.loader.progress_msg.connect(self._on_loader_progress)
        self.loader.finished_ok.connect(self._on_load_done)
        self.loader.finished_err.connect(self._on_load_error)
        self.loader.start()

    def _on_loader_progress(self, msg: str):
        self.proc_lbl.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_load_done(self, mesh, cusp_pts):
        self._hide_overlay()
        self.mesh     = mesh
        self.cusp_pts = cusp_pts
        self.selected = []

        self.cusp_count_lbl.setText(str(len(cusp_pts)))
        self.sel_lbl.setText("0 / 3 cusps selected")
        self.dev_lbl.setText("—")
        self.monson_lbl.setText(
            "Select 3 cusp tips in the 3-D view\n(hover + press P)."
        )

        self.status_bar.showMessage(
            f"Loaded — {len(mesh.points):,} vertices, "
            f"{len(cusp_pts)} cusp tip(s) detected."
        )

        self.open_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        for b in self._view_buttons:
            b.setEnabled(True)

        self._build_plotter()

    def _on_load_error(self, err: str):
        self._hide_overlay()
        self.status_bar.showMessage(f"Error: {err}")
        self.open_btn.setEnabled(True)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Load Error", f"Could not load mesh:\n{err}")

    # ------------------------------------------------------------------
    # Camera view presets
    # ------------------------------------------------------------------
    def _set_view(self, view: str):
        if self.plotter is None or self.mesh is None:
            return
        presets = {
            "front":  dict(position=(0, -1, 0),  viewup=(0, 0, 1)),
            "back":   dict(position=(0,  1, 0),  viewup=(0, 0, 1)),
            "top":    dict(position=(0,  0, 1),  viewup=(0, 1, 0)),
            "bottom": dict(position=(0,  0, -1), viewup=(0, 1, 0)),
            "left":   dict(position=(-1, 0, 0),  viewup=(0, 0, 1)),
            "right":  dict(position=( 1, 0, 0),  viewup=(0, 0, 1)),
            "iso":    dict(position=(1, -1, 1),   viewup=(0, 0, 1)),
        }
        if view not in presets:
            return
        p = presets[view]
        bounds = self.mesh.bounds          # (xmin,xmax,ymin,ymax,zmin,zmax)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        default_dist = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ) * 2.5

        # Retain current zoom level (camera distance to focal point)
        curr_dist = self.plotter.camera.distance
        dist = curr_dist if (curr_dist is not None and curr_dist > 0) else default_dist

        pos_dir = np.array(p["position"], dtype=float)
        norm = np.linalg.norm(pos_dir)
        if norm > 0:
            pos_dir = pos_dir / norm

        self.plotter.camera.focal_point = (cx, cy, cz)
        self.plotter.camera.position    = (cx + pos_dir[0] * dist,
                                           cy + pos_dir[1] * dist,
                                           cz + pos_dir[2] * dist)
        self.plotter.camera.up          = p["viewup"]
        self.plotter.render()

    # ------------------------------------------------------------------
    # Build / refresh the VTK plotter
    # ------------------------------------------------------------------
    def _build_plotter(self):
        # Remove old plotter if exists
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

        # Remove placeholder
        self.placeholder.hide()

        # Create embedded plotter
        self.plotter = BackgroundPlotter(show=False)
        self.plotter.set_background("#0d1117")

        # Add mesh
        self.plotter.add_mesh(self.mesh, color="#b0bec5", opacity=0.92,
                               smooth_shading=True)

        # Add green cusp markers
        for c in self.cusp_pts:
            self.plotter.add_mesh(
                pv.Sphere(radius=0.8, center=c), color="#68d391"
            )

        # Enable trackball camera style (supports scroll wheel zoom & pan)
        self.plotter.enable_trackball_style()

        # Enable picking — left_clicking=False keeps mouse buttons free
        # for camera navigation (rotate / zoom / pan).
        # Users pick a cusp by hovering over it and pressing the P key.
        try:
            self.plotter.enable_point_picking(
                callback=self._pick_callback,
                show_message=False,
                show_point=True,
                tolerance=0.025,
                left_clicking=False,   # ← do NOT hijack mouse events
            )
        except TypeError:
            # Older PyVista versions that don't accept left_clicking
            self.plotter.enable_point_picking(
                callback=self._pick_callback,
                show_message=False,
                show_point=True,
                tolerance=0.025,
            )

        # Embed Qt widget
        vtk_lay = self.vtk_container.layout()
        vtk_lay.addWidget(self.plotter)

    # ------------------------------------------------------------------
    # Point picking
    # ------------------------------------------------------------------
    def _pick_callback(self, point):
        if point is None:
            return
        self.selected.append(np.array(point))
        n = len(self.selected)
        self.sel_lbl.setText(f"{n} / 3 cusps selected")

        self.plotter.add_mesh(
            pv.Sphere(radius=1.2, center=point), color="#f6e05e"
        )

        if n == 3:
            self._compute_and_render_monson()

    def _compute_and_render_monson(self):
        pts    = np.array(self.selected)
        center = compute_monson(pts)

        # Draw Monson sphere
        # ambient=1.0 / diffuse=0.0: render in true flat colour so VTK
        # lighting doesn't darken the sphere against the dark background.
        sphere = pv.Sphere(radius=MONSON_RADIUS, center=center,
                           theta_resolution=120, phi_resolution=120)
        self.plotter.add_mesh(sphere, color="#7cced9", opacity=0.10,
                              ambient=1.0, diffuse=0.0, specular=0.0)

        # Deviation analysis
        lines = []
        for cusp in self.cusp_pts:
            deviation = np.linalg.norm(cusp - center) - MONSON_RADIUS
            is_above = deviation > 0
            color = "#fc8181" if is_above else "#63b3ed"
            badge_color = "#dc2626" if is_above else "#2563eb"
            status = "above" if is_above else "below"

            self.plotter.add_mesh(
                pv.Sphere(radius=0.9, center=cusp), color=color
            )
            self.plotter.add_point_labels(
                [cusp],
                [f"{deviation:+.2f} mm"],
                point_size=16,
                text_color="#ffffff",
                shape_color=badge_color,
                shape_opacity=0.9,
                font_size=11,
                bold=True,
            )
            lines.append(f"  {status}  {deviation:+.2f} mm")

        self.monson_lbl.setText(
            f"Centre: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}\n"
            f"Radius: {MONSON_RADIUS} mm"
        )
        self.dev_lbl.setText("\n".join(lines) if lines else "No data")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset(self):
        if self.mesh is None:
            return
        self.selected = []
        self.sel_lbl.setText("0 / 3 cusps selected")
        self.dev_lbl.setText("—")
        self.monson_lbl.setText(
            "Select 3 cusp tips in the 3-D view\n(hover + press P)."
        )
        # Rebuild plotter from scratch (clean slate)
        self._build_plotter()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("DOPA")
    app.setStyle("Fusion")

    win = DOPAWindow()
    win.show()                      # ← window is immediately visible


    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
